//! The tests for this module.
//!
//! They sit here together rather than beside each file because they exercise
//! the module as a whole: a fetch writes the cache, hashes what it wrote, and
//! unpacks what it downloaded, and one test often touches all three.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::hash::{collect, sha256, sha256_of_dir, sha256_of_file};
use super::http::{
    fetch_timeout, fetch_timeout_secs, is_api_host, redirect_follow, same_origin,
    DEFAULT_FETCH_TIMEOUT_SECS, MAX_REDIRECTS,
};
use super::source::*;
use super::unpack::{extract_archive, extract_archive_with, is_archive_name, safe_mode};

    #[test]
    fn a_github_reference_parses_into_its_parts() {
        let source = Source::parse("github:owner/repo@skills/pdf@v1.2.0").unwrap();
        assert_eq!(
            source,
            Source::Github {
                owner: "owner".into(),
                repo: "repo".into(),
                path: "skills/pdf".into(),
                git_ref: "v1.2.0".into(),
            }
        );
    }

    #[test]
    fn a_github_reference_may_omit_the_path_and_the_ref() {
        assert_eq!(
            Source::parse("github:owner/repo").unwrap(),
            Source::Github {
                owner: "owner".into(),
                repo: "repo".into(),
                path: ".".into(),
                git_ref: "HEAD".into(),
            }
        );
        // Only a ref.
        assert_eq!(
            Source::parse("github:owner/repo@v2").unwrap(),
            Source::Github {
                owner: "owner".into(),
                repo: "repo".into(),
                path: "v2".into(),
                git_ref: "HEAD".into(),
            }
        );
    }

    #[test]
    fn the_ref_is_taken_from_the_right() {
        // A path containing `@` must not be mistaken for the ref.
        let source = Source::parse("github:owner/repo@dir/we@ird/file@v1").unwrap();
        match source {
            Source::Github { path, git_ref, .. } => {
                assert_eq!(path, "dir/we@ird/file");
                assert_eq!(git_ref, "v1");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_trailing_git_suffix_is_dropped() {
        match Source::parse("github:owner/repo.git@v1").unwrap() {
            Source::Github { repo, .. } => assert_eq!(repo, "repo"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_other_forms_parse() {
        assert_eq!(
            Source::parse("https://example.com/a.md").unwrap(),
            Source::Url {
                url: "https://example.com/a.md".into()
            }
        );
        assert_eq!(
            Source::parse("builtin:search").unwrap(),
            Source::Builtin {
                name: "search".into()
            }
        );
        assert_eq!(
            Source::parse("dylib:files/x.dll").unwrap(),
            Source::Local {
                path: "files/x.dll".into()
            }
        );
    }

    #[test]
    fn nonsense_is_refused_with_the_accepted_forms() {
        for bad in [
            "",
            "   ",
            "github:",
            "github:owner",
            "builtin:",
            "no-colon-here",
            ":empty-scheme",
        ] {
            let error = Source::parse(bad).expect_err(bad);
            let text = format!("{error:#}");
            assert!(
                text.contains("github") || text.contains("names no plugin"),
                "{bad:?} produced: {text}"
            );
        }
    }

    #[test]
    fn a_path_inside_the_pack_parses_as_local() {
        // The offline shape pack-format documents ("相对路径"): a pack shipping
        // its own skills writes them without a scheme. A colon-bearing string
        // was already claimed by the scheme branch, so this cannot swallow a
        // URL or `C:\...`; a bare word still fails.
        for raw in ["./skills/pdf", "skills/pdf", "..\\shared\\skills"] {
            assert_eq!(
                Source::parse(raw).unwrap(),
                Source::Local {
                    path: raw.to_string()
                },
                "{raw}"
            );
        }
        assert!(Source::parse("no-colon-here").is_err(), "a word is not a path");
    }

    #[test]
    fn an_unknown_scheme_is_kept_rather_than_refused() {
        // Which schemes exist is not the format's business: a deployment may
        // serve packs from an internal registry or a git host that is not
        // GitHub, and the set of names cannot be enumerated in advance.
        let source = Source::parse("gitlab:owner/repo@path@v1").unwrap();
        assert_eq!(
            source,
            Source::Custom {
                scheme: "gitlab".into(),
                rest: "owner/repo@path@v1".into(),
            }
        );
        assert!(source.is_remote(), "most custom schemes fetch over the network");
    }

    #[test]
    fn a_custom_scheme_reports_a_clean_error_when_no_handler_is_registered() {
        let fetcher = Fetcher::at(scratch("noscheme")).unwrap();
        let source = Source::parse("gitlab:owner/repo@x@v1").unwrap();
        let error = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async { fetcher.fetch(&source, None).await })
            .expect_err("must refuse");
        let text = format!("{error:#}");
        assert!(text.contains("gitlab"), "{text}");
        assert!(
            text.contains("github"),
            "it should list what is registered: {text}"
        );
    }

    #[tokio::test]
    async fn a_registered_scheme_is_fetched_through_its_handler() {
        // The extension point: a deployment serves packs from wherever it keeps
        // them, and the format learns nothing about that place.
        let served = scratch("served");
        std::fs::create_dir_all(served.join("myskill")).unwrap();
        std::fs::write(served.join("myskill/SKILL.md"), "a skill").unwrap();

        let mut fetcher = Fetcher::at(scratch("custom-cache")).unwrap();
        fetcher.register_scheme(Arc::new(DirectoryScheme::new("internal", &served)));
        assert!(fetcher.schemes().contains(&"internal".to_string()));

        let source = Source::parse("internal:myskill").unwrap();
        let fetched = fetcher.fetch(&source, None).await.unwrap();
        assert!(fetched.path.join("SKILL.md").is_file());
        assert_eq!(
            std::fs::read_to_string(fetched.path.join("SKILL.md")).unwrap(),
            "a skill"
        );
    }

    #[test]
    fn only_remote_sources_need_the_network() {
        assert!(Source::parse("github:o/r@x@v").unwrap().is_remote());
        assert!(Source::parse("https://x/y").unwrap().is_remote());
        assert!(!Source::parse("builtin:search").unwrap().is_remote());
        assert!(!Source::parse("dylib:x.dll").unwrap().is_remote());
    }

    #[test]
    fn a_directory_hash_is_stable_regardless_of_creation_order() {
        let a = scratch("order-a");
        let b = scratch("order-b");
        for dir in [&a, &b] {
            std::fs::create_dir_all(dir.join("sub")).unwrap();
        }
        // Same content, written in opposite orders.
        std::fs::write(a.join("one.txt"), "first").unwrap();
        std::fs::write(a.join("sub/two.txt"), "second").unwrap();
        std::fs::write(b.join("sub/two.txt"), "second").unwrap();
        std::fs::write(b.join("one.txt"), "first").unwrap();

        assert_eq!(sha256_of_dir(&a).unwrap(), sha256_of_dir(&b).unwrap());
    }

    #[test]
    fn a_directory_hash_changes_with_content() {
        let dir = scratch("content");
        std::fs::write(dir.join("f.txt"), "before").unwrap();
        let before = sha256_of_dir(&dir).unwrap();
        std::fs::write(dir.join("f.txt"), "after").unwrap();
        assert_ne!(before, sha256_of_dir(&dir).unwrap());
    }

    #[test]
    fn a_directory_hash_notices_a_renamed_file() {
        // The path is part of the hash, not only the bytes.
        let a = scratch("rename-a");
        let b = scratch("rename-b");
        std::fs::write(a.join("x.txt"), "same").unwrap();
        std::fs::write(b.join("y.txt"), "same").unwrap();
        assert_ne!(sha256_of_dir(&a).unwrap(), sha256_of_dir(&b).unwrap());
    }

    #[test]
    fn a_file_hash_matches_the_byte_hash() {
        let dir = scratch("file");
        let path = dir.join("f.bin");
        std::fs::write(&path, [1u8, 2, 3]).unwrap();
        assert_eq!(sha256_of_file(&path).unwrap(), sha256(&[1, 2, 3]));
    }

    #[test]
    fn the_known_sha256_of_a_known_input_is_stable() {
        // Pinning one vector catches an accidental algorithm change, which
        // would silently invalidate every published pack hash.
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_cache_lookup_refuses_a_hash_that_is_a_path() {
        let fetcher = Fetcher::at(scratch("cache")).unwrap();
        assert!(fetcher.cache_hit("../../etc").is_none());
        assert!(fetcher.cache_hit("").is_none());
        assert!(fetcher.cache_hit("not-hex").is_none());
    }

    #[test]
    fn a_cache_lookup_finds_what_is_there() {
        let root = scratch("hit");
        let hash = "a".repeat(64);
        std::fs::create_dir_all(root.join(&hash)).unwrap();
        let fetcher = Fetcher::at(&root).unwrap();
        assert_eq!(fetcher.cache_hit(&hash), Some(root.join(&hash)));
        assert!(fetcher.cache_hit(&"b".repeat(64)).is_none());
    }

    #[tokio::test]
    async fn a_cached_hash_is_reused_without_touching_the_network() {
        let root = scratch("reuse");
        let hash = "c".repeat(64);
        std::fs::create_dir_all(root.join(&hash)).unwrap();
        let fetcher = Fetcher::at(&root).unwrap();

        // A source that cannot resolve, proving the cache short-circuits it.
        let source = Source::parse("github:nobody/nothing@x@y").unwrap();
        let fetched = fetcher.fetch(&source, Some(&hash)).await.unwrap();

        assert!(fetched.cached);
        assert_eq!(fetched.path, root.join(&hash));
    }

    #[tokio::test]
    async fn a_non_remote_source_is_refused_by_fetch() {
        let fetcher = Fetcher::at(scratch("local")).unwrap();
        let source = Source::parse("builtin:search").unwrap();
        let error = fetcher.fetch(&source, None).await.expect_err("must refuse");
        assert!(format!("{error:#}").contains("not a remote source"));
    }

    #[test]
    fn the_platform_tag_is_one_the_format_knows() {
        assert!(PLATFORMS.contains(&platform_tag()), "{}", platform_tag());
    }

    #[test]
    fn a_source_describes_itself() {
        assert_eq!(
            Source::parse("github:o/r@p@v").unwrap().describe(),
            "o/r@p@v"
        );
        assert_eq!(Source::parse("builtin:x").unwrap().describe(), "builtin:x");
    }

    #[test]
    fn archive_recognition_looks_at_the_name() {
        assert!(is_archive_name("cua-driver-rs-0.28.0-windows-x86_64-binary.zip"));
        assert!(is_archive_name("https://host/x/driver.tar.gz"));
        assert!(is_archive_name("driver.tgz?download=1"));
        assert!(!is_archive_name("driver.tar"));
        assert!(!is_archive_name("driver"));
    }

    #[test]
    fn a_zip_unpacks_into_the_directory_it_is_given() {
        let dir = scratch("extract-zip");
        let archive = dir.join("bundle.zip");
        {
            let file = std::fs::File::create(&archive).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default();
            writer.start_file("tool.exe", options).unwrap();
            writer.write_all(b"binary").unwrap();
            writer.start_file("docs/readme.txt", options).unwrap();
            writer.write_all(b"notes").unwrap();
            writer.finish().unwrap();
        }

        let into = dir.join("out");
        extract_archive(&archive, &into).unwrap();
        assert_eq!(std::fs::read(into.join("tool.exe")).unwrap(), b"binary");
        assert_eq!(
            std::fs::read_to_string(into.join("docs/readme.txt")).unwrap(),
            "notes"
        );
    }

    #[test]
    fn a_zip_entry_that_escapes_is_refused() {
        let dir = scratch("extract-escape");
        let archive = dir.join("evil.zip");
        {
            let file = std::fs::File::create(&archive).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            writer
                .start_file("../outside.txt", zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(b"escaped").unwrap();
            writer.finish().unwrap();
        }

        let error = extract_archive(&archive, &dir.join("out")).expect_err("must refuse");
        assert!(
            format!("{error:#}").contains("escapes the archive"),
            "{error:#}"
        );
        assert!(!dir.join("outside.txt").exists(), "nothing was written outside");
    }

    #[test]
    fn a_tgz_unpacks_and_keeps_the_executable_bit() {
        let dir = scratch("extract-tgz");
        let archive = dir.join("bundle.tar.gz");
        write_tgz(&archive, "bin/tool", 0o755, b"script");

        let into = dir.join("out");
        extract_archive(&archive, &into).unwrap();
        assert_eq!(std::fs::read(into.join("bin/tool")).unwrap(), b"script");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(into.join("bin/tool"))
                .unwrap()
                .permissions()
                .mode();
            assert_ne!(mode & 0o111, 0, "the executable bit survives: {mode:o}");
        }
    }

    #[test]
    fn a_tgz_entry_that_escapes_is_refused() {
        // The same refusal for tar as for zip: a `..` name aborts before a
        // single byte is written.
        let dir = scratch("extract-tgz-escape");
        let archive = dir.join("evil.tar.gz");
        write_tgz_raw_name(&archive, "../outside.txt", 0o644, b"bad");

        let error = extract_archive(&archive, &dir.join("out")).expect_err("must refuse");
        assert!(
            format!("{error:#}").contains("escapes the archive"),
            "{error:#}"
        );
        assert!(!dir.join("outside.txt").exists());
    }

    #[test]
    fn a_name_that_is_not_an_archive_says_so() {
        let dir = scratch("extract-unknown");
        let plain = dir.join("bundle.rar");
        std::fs::write(&plain, b"not an archive").unwrap();
        let error = extract_archive(&plain, &dir.join("out")).expect_err("must refuse");
        assert!(format!("{error:#}").contains("expected .zip"), "{error:#}");
    }

    #[tokio::test]
    async fn a_single_file_fetch_reuses_the_cache_keyed_by_the_file_hash() {
        // The cache hit answers before any network call, which is what makes
        // a second install of the same pack work with no connection.
        let root = scratch("file-fetch");
        let bytes = b"release asset";
        let hash = sha256(bytes);
        std::fs::create_dir_all(root.join(&hash)).unwrap();
        std::fs::write(root.join(&hash).join("asset.zip"), bytes).unwrap();

        let fetcher = Fetcher::at(&root).unwrap();
        let fetched = fetcher
            .fetch_file("https://example.invalid/asset.zip", &hash)
            .await
            .unwrap();
        assert!(fetched.cached, "the seeded entry answers the fetch");
        assert_eq!(fetched.path, root.join(&hash));
        assert_eq!(
            std::fs::read(fetched.path.join("asset.zip")).unwrap(),
            bytes
        );
    }

    /// A loopback HTTP server answering every request with `body`.
    ///
    /// With `declared` the response carries a `content-length`; without it the
    /// body runs to the end of the connection, which is how a chunked or
    /// length-lying server reaches the client. Returns the base URL and a count
    /// of the requests actually served.
    async fn serving(
        body: Vec<u8>,
        declared: bool,
    ) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&hits);

        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let body = body.clone();
                let counter = Arc::clone(&counter);
                tokio::spawn(async move {
                    // Drain the request before answering: a close that races a
                    // client still writing turns every test into a flake.
                    let mut head = Vec::new();
                    let mut byte = [0u8; 1];
                    loop {
                        match socket.read(&mut byte).await {
                            Ok(0) | Err(_) => return,
                            Ok(_) => {
                                head.push(byte[0]);
                                if head.ends_with(b"\r\n\r\n") {
                                    break;
                                }
                            }
                        }
                    }
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    let mut response = b"HTTP/1.1 200 OK\r\nconnection: close\r\n".to_vec();
                    if declared {
                        response.extend_from_slice(
                            format!("content-length: {}\r\n", body.len()).as_bytes(),
                        );
                    }
                    response.extend_from_slice(b"\r\n");
                    response.extend_from_slice(&body);
                    let _ = socket.write_all(&response).await;
                    let _ = socket.flush().await;
                    // Hold the socket a moment so the client's read sees an
                    // orderly end of body.
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                });
            }
        });

        (format!("http://{addr}"), hits)
    }

    #[tokio::test]
    async fn a_download_whose_declared_length_is_over_the_ceiling_is_refused() {
        // Before the fix, nothing bounded a download at all: one URL that
        // answers 4 GiB was an out-of-memory kill rather than an error.
        let (base, hits) = serving(vec![b'z'; 4096], true).await;
        let fetcher = Fetcher::at(scratch("declared-ceiling")).unwrap();

        let error = fetcher
            .get_bytes_capped(&base, 1024)
            .await
            .expect_err("a declared body over the ceiling is refused");
        assert!(format!("{error:#}").contains("ceiling"), "{error:#}");
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "a size refusal is not a transport failure, so it is not retried"
        );
    }

    #[tokio::test]
    async fn a_download_without_a_declared_length_is_capped_as_it_streams() {
        // The same ceiling for a server that declares nothing: the running
        // total has to trip, not the header.
        let (base, hits) = serving(vec![b'z'; 8192], false).await;
        let fetcher = Fetcher::at(scratch("streaming-ceiling")).unwrap();

        let error = fetcher
            .get_bytes_capped(&base, 1024)
            .await
            .expect_err("an undeclared body is still bounded");
        assert!(format!("{error:#}").contains("ceiling"), "{error:#}");
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn a_download_within_the_ceiling_arrives_whole() {
        // The happy path, so the cap is proved not to eat a normal body.
        let payload: Vec<u8> = (0..600u32).map(|index| (index % 251) as u8).collect();
        let (base, _hits) = serving(payload.clone(), true).await;
        let fetcher = Fetcher::at(scratch("whole")).unwrap();

        let bytes = fetcher
            .get_bytes_capped(&base, 1024 * 1024)
            .await
            .expect("the download completes");
        assert_eq!(bytes, payload, "byte for byte, one copy");
    }

    #[tokio::test]
    async fn a_connection_dropped_before_the_answer_is_retried() {
        // The fetch path used to have no retry at all, while the model client
        // had one: the same peer closing a connection ended a download and so
        // an install. A GET has no side effects, so asking again is safe.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&hits);
        let payload = b"the pack contents".to_vec();
        let served = payload.clone();

        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut answered = false;
            while let Ok((mut socket, _)) = listener.accept().await {
                let seen = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if !answered && seen == 0 {
                    // Hang up without writing a single response byte. The
                    // request itself is already in the socket buffer, so the
                    // client sees a connection that went away mid-answer.
                    drop(socket);
                    continue;
                }
                answered = true;

                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                loop {
                    match socket.read(&mut byte).await {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            head.push(byte[0]);
                            if head.ends_with(b"\r\n\r\n") {
                                break;
                            }
                        }
                    }
                }
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    served.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.write_all(&served).await;
                let _ = socket.flush().await;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        });

        let fetcher = Fetcher::at(scratch("dropped")).unwrap();
        let bytes = fetcher
            .get_bytes_capped(&format!("http://{addr}/asset.bin"), 1024 * 1024)
            .await
            .expect("the retry recovers from a dropped connection");
        assert_eq!(bytes, payload);
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "one dropped connection, one retry"
        );
    }

    #[tokio::test]
    async fn a_refusal_is_not_retried() {
        // A 4xx is the server's final word on the request: retrying it spends
        // the round trip twice for the same answer, and turns a missing file
        // into three.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&hits);
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            while let Ok((mut socket, _)) = listener.accept().await {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let _ = socket
                    .write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                    .await;
                let _ = socket.flush().await;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        });

        let fetcher = Fetcher::at(scratch("refused")).unwrap();
        let error = fetcher
            .get_bytes(&format!("http://{addr}/missing.bin"))
            .await
            .expect_err("a 404 is an error");
        assert!(format!("{error:#}").contains("404"), "{error:#}");
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn the_fetch_timeout_follows_its_environment_variable() {
        // The old code had 120 seconds hardcoded and no way to say otherwise;
        // a pack mirror on a slow link could not be fetched at all.
        assert_eq!(fetch_timeout_secs(Some("45")), 45);
        assert_eq!(fetch_timeout_secs(Some(" 45 ")), 45, "padded is still a number");
        assert_eq!(fetch_timeout_secs(None), DEFAULT_FETCH_TIMEOUT_SECS);
        // A typo falls back rather than producing a timeout that always fires.
        assert_eq!(fetch_timeout_secs(Some("soon")), DEFAULT_FETCH_TIMEOUT_SECS);
        assert_eq!(fetch_timeout_secs(Some("0")), DEFAULT_FETCH_TIMEOUT_SECS);
        assert!(fetch_timeout().as_secs() >= 1, "never an instant timeout");
    }

    /// A `.zip` holding one entry of `size` zero bytes: a few kilobytes on
    /// disk, arbitrarily large once expanded. This is the shape of a
    /// decompression bomb in the smallest useful form.
    fn write_zip_of_zeros(into: &Path, name: &str, size: usize) {
        let file = std::fs::File::create(into).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        writer
            .start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(&vec![0u8; size]).unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn an_archive_that_expands_past_the_entry_ceiling_is_refused() {
        // A small archive that unpacks to something enormous is exactly what a
        // compressor is for, so an untrusted one has to be refused on the
        // expanded size rather than believed about it.
        let dir = scratch("zip-bomb");
        let archive = dir.join("bomb.zip");
        write_zip_of_zeros(&archive, "payload.bin", 4 * 1024 * 1024);
        assert!(
            std::fs::metadata(&archive).unwrap().len() < 512 * 1024,
            "the archive on disk is small; only its expansion is not"
        );

        let out = dir.join("out");
        let limits = UnpackLimits {
            entries: 16,
            entry_bytes: 64 * 1024,
            total_bytes: 1024 * 1024,
        };
        let error =
            extract_archive_with(&archive, &out, limits).expect_err("an expanding archive is refused");

        let text = format!("{error:#}");
        assert!(text.contains("payload.bin"), "the entry is named: {text}");
        assert!(text.contains("ceiling"), "the ceiling is named: {text}");
        assert!(
            !out.join("payload.bin").exists(),
            "nothing of the entry was written"
        );
    }

    #[test]
    fn an_archive_with_too_many_entries_is_refused() {
        // The cheap half of the same attack: many tiny entries, refused before
        // any of them is read.
        let dir = scratch("zip-many");
        let archive = dir.join("many.zip");
        {
            let file = std::fs::File::create(&archive).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            for index in 0..8 {
                writer
                    .start_file(format!("f{index}.txt"), zip::write::SimpleFileOptions::default())
                    .unwrap();
                writer.write_all(b"x").unwrap();
            }
            writer.finish().unwrap();
        }

        let limits = UnpackLimits {
            entries: 4,
            entry_bytes: 1024,
            total_bytes: 4096,
        };
        let error = extract_archive_with(&archive, &dir.join("out"), limits)
            .expect_err("an archive with too many entries is refused");
        let text = format!("{error:#}");
        assert!(
            text.contains("more than 4 entries"),
            "the entry ceiling is named: {text}"
        );
    }

    #[test]
    fn an_archive_that_expands_past_the_total_ceiling_is_refused() {
        // Each entry fits on its own; together they do not.
        let dir = scratch("zip-total");
        let archive = dir.join("total.zip");
        {
            let file = std::fs::File::create(&archive).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            for index in 0..4 {
                writer
                    .start_file(format!("p{index}.bin"), zip::write::SimpleFileOptions::default())
                    .unwrap();
                writer.write_all(&vec![b'z'; 4096]).unwrap();
            }
            writer.finish().unwrap();
        }

        let limits = UnpackLimits {
            entries: 16,
            entry_bytes: 8192,
            total_bytes: 8192,
        };
        let error = extract_archive_with(&archive, &dir.join("out"), limits)
            .expect_err("the whole archive is capped");
        let text = format!("{error:#}");
        assert!(
            text.contains("one archive"),
            "the total ceiling is named: {text}"
        );
    }

    #[test]
    fn a_normal_archive_is_unpacked_under_the_default_ceilings() {
        // The ceilings must not get in the way of the thing they exist for: a
        // real archive whose entries are ordinary files.
        let dir = scratch("zip-normal");
        let archive = dir.join("normal.zip");
        {
            let file = std::fs::File::create(&archive).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            writer
                .start_file("readme.txt", zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(b"hello").unwrap();
            writer.finish().unwrap();
        }
        let out = dir.join("out");
        extract_archive(&archive, &out).expect("a small archive is unpacked");
        assert_eq!(std::fs::read_to_string(out.join("readme.txt")).unwrap(), "hello");
    }

    #[test]
    fn an_archive_cannot_hand_out_a_setuid_binary() {
        // Mode bits come from the archive's author, and a setuid binary dropped
        // into a plugin directory is a privilege escalation the install never
        // asked for.
        assert_eq!(safe_mode(0o4755), 0o755, "the setuid bit is dropped");
        assert_eq!(safe_mode(0o2755), 0o755, "so is setgid");
        assert_eq!(safe_mode(0o6755), 0o755, "both of them");
        assert_eq!(safe_mode(0o7777), 0o755, "and the sticky bit");
        assert_eq!(
            safe_mode(0o4644),
            0o644,
            "setuid without an execute bit is just a file"
        );
        assert_eq!(safe_mode(0o600), 0o644, "a private file becomes readable");
        assert_eq!(safe_mode(0o100), 0o755, "any execute bit means 0o755");
    }

    #[cfg(unix)]
    #[test]
    fn an_extracted_binary_is_not_setuid() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("setuid-unpack");
        let archive = dir.join("tool.tgz");
        write_tgz(&archive, "tool", 0o4755, b"binary");

        let into = dir.join("out");
        extract_archive(&archive, &into).unwrap();

        let mode = std::fs::metadata(into.join("tool"))
            .unwrap()
            .permissions()
            .mode()
            & 0o7777;
        assert_eq!(mode, 0o755, "the setuid bit must not survive the unpack");
    }

    #[test]
    fn a_redirect_that_downgrades_to_http_is_refused() {
        // Following this one hands the first host's credential to whoever is
        // listening on the plain-text side of the hop.
        let chain = vec![reqwest::Url::parse("https://api.github.com/repos/o/r/zipball").unwrap()];
        let error = redirect_follow(&chain, "http://mirror.test/x.zip").expect_err("downgrade refused");
        let text = format!("{error:#}");
        assert!(text.contains("downgrades"), "{text}");
        assert!(text.contains("mirror.test"), "the hop is named: {text}");
    }

    #[test]
    fn a_redirect_chain_is_bounded() {
        let mut chain = vec![reqwest::Url::parse("https://host.test/0").unwrap()];
        for hop in 1..=5 {
            chain.push(reqwest::Url::parse(&format!("https://host.test/{hop}")).unwrap());
        }
        let error = redirect_follow(&chain, "/next").expect_err("the sixth hop is refused");
        assert!(
            format!("{error:#}").contains("5 times"),
            "the bound is named: {error:#}"
        );
    }

    #[test]
    fn a_relative_location_resolves_against_the_hop_it_came_from() {
        let chain = vec![reqwest::Url::parse("https://host.test/a/b.zip").unwrap()];
        assert_eq!(
            redirect_follow(&chain, "c.zip").unwrap().as_str(),
            "https://host.test/a/c.zip"
        );
        assert_eq!(
            redirect_follow(&chain, "/d.zip").unwrap().as_str(),
            "https://host.test/d.zip"
        );
    }

    #[test]
    fn a_credential_stays_on_the_origin_it_was_issued_for() {
        let api = reqwest::Url::parse("https://api.github.com/repos/o/r/git/trees/main").unwrap();
        let also_api = reqwest::Url::parse("https://api.github.com/raw/x").unwrap();
        let codeload = reqwest::Url::parse("https://codeload.github.com/o/r/zip").unwrap();
        let plain = reqwest::Url::parse("http://api.github.com/x").unwrap();

        assert!(is_api_host(&api));
        assert!(same_origin(&api, &also_api));
        assert!(!same_origin(&api, &codeload), "another host is another origin");
        assert!(!same_origin(&api, &plain), "and another scheme is too");

        // A path that merely spells the API host is not the API host: the old
        // substring test handed the token to this URL.
        let impostor =
            reqwest::Url::parse("https://elsewhere.test/api.github.com/x").unwrap();
        assert!(!is_api_host(&impostor), "{impostor}");
    }

    #[tokio::test]
    async fn a_cross_origin_redirect_is_followed_without_a_credential() {
        // GitHub answers a release-asset request on one host with a redirect to
        // another, so refusing every cross-origin hop would break downloads.
        // What must not happen is a credential crossing with it — and no
        // credential is attached here, because the first host has none.
        let payload = b"release asset".to_vec();
        let (destination, destination_hits) = serving(payload.clone(), true).await;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let target = format!("{destination}/asset.zip");
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            while let Ok((mut socket, _)) = listener.accept().await {
                let target = target.clone();
                tokio::spawn(async move {
                    let mut head = Vec::new();
                    let mut byte = [0u8; 1];
                    loop {
                        match socket.read(&mut byte).await {
                            Ok(0) | Err(_) => return,
                            Ok(_) => {
                                head.push(byte[0]);
                                if head.ends_with(b"\r\n\r\n") {
                                    break;
                                }
                            }
                        }
                    }
                    let response = format!(
                        "HTTP/1.1 302 Found\r\nlocation: {target}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.flush().await;
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                });
            }
        });

        let fetcher = Fetcher::at(scratch("redirect-cross")).unwrap();
        let bytes = fetcher
            .get_bytes_capped(&format!("http://{addr}/asset.zip"), 1024 * 1024)
            .await
            .expect("the redirect is followed");
        assert_eq!(bytes, payload);
        assert_eq!(
            destination_hits.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the payload came from the redirect target"
        );
    }

    #[tokio::test]
    async fn a_redirect_loop_is_bounded() {
        // The client's own policy would follow ten hops before giving up, and
        // every one of them would carry whatever header the first one had.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&hits);
        let location = format!("http://{addr}/again");

        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            while let Ok((mut socket, _)) = listener.accept().await {
                let location = location.clone();
                let counter = Arc::clone(&counter);
                tokio::spawn(async move {
                    let mut head = Vec::new();
                    let mut byte = [0u8; 1];
                    loop {
                        match socket.read(&mut byte).await {
                            Ok(0) | Err(_) => return,
                            Ok(_) => {
                                head.push(byte[0]);
                                if head.ends_with(b"\r\n\r\n") {
                                    break;
                                }
                            }
                        }
                    }
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let response = format!(
                        "HTTP/1.1 302 Found\r\nlocation: {location}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.flush().await;
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                });
            }
        });

        let fetcher = Fetcher::at(scratch("redirect-loop")).unwrap();
        let error = fetcher
            .get_bytes_capped(&format!("http://{addr}/start"), 1024 * 1024)
            .await
            .expect_err("an endless redirect is an error");
        assert!(
            format!("{error:#}").contains("5 times"),
            "the bound is named: {error:#}"
        );
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::Relaxed),
            MAX_REDIRECTS + 1,
            "the first request plus the five hops it allowed"
        );
    }

    /// One small `.tar.gz` holding a single file, built in the test rather
    /// than kept as a fixture.
    fn write_tgz(into: &Path, name: &str, mode: u32, contents: &[u8]) {
        let file = std::fs::File::create(into).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(mode);
        header.set_cksum();
        builder
            .append_data(&mut header, name, contents)
            .unwrap();
        builder.finish().unwrap();
        builder.into_inner().unwrap().finish().unwrap();
    }

    /// A `.tar.gz` whose entry name is written straight into the header
    /// bytes: `tar::Builder` refuses to *set* a `..` path, so an archive
    /// that actually carries one has to be laid down by hand.
    fn write_tgz_raw_name(into: &Path, name: &str, mode: u32, contents: &[u8]) {
        let file = std::fs::File::create(into).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(mode);
        {
            let bytes = header.as_mut_bytes();
            for (index, byte) in name.as_bytes().iter().enumerate() {
                bytes[index] = *byte;
            }
        }
        header.set_cksum();
        builder.append(&header, contents).unwrap();
        builder.finish().unwrap();
        builder.into_inner().unwrap().finish().unwrap();
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("ngu-fetch-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
