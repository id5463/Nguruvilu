// delegate 包：把一个子任务交给另一个独立的 ngu 进程。
//
// # 为什么是进程而不是内核里的一段代码
//
// 它以前是 `src/tools/subagent.rs`，962 行，用了内核里 12 种类型
// （ModelAccess、TurnConfig、Message、ModelHandle…）—— 本质上是在进程内
// 重新组装一个 Agent。而 `ngu -p` 已经是一个完整的 agent。再组装一遍就是
// 重复，而且重复出来的那个只能是"这一种"子代理。
//
// 做成进程之后：
//
//   * 子代理有自己的进程、自己的 session、自己的失败，主代理不受影响
//   * 可以用 --bare 不带人格、--packs none 不装包（162 毫秒启动）
//   * 换一种子代理就是换一个包，不用改内核
//
// # 上下文是空的
//
// `ngu -p` 起的是新会话，看不到主对话说过什么。子任务必须自包含：说清要
// 做什么、在哪里、什么算做完。这是它的价值所在 —— 一个大的调查挡在主对话
// 外面，主对话只收到结论，不是那些文件的原文。
//
// # 递归
//
// 子代理自己也能调 delegate，所以深度要靠环境变量传下去：每个进程都知道
// 自己在第几层，到上限就拒绝。没有这个，一个指令能开出一棵无限深的树，
// 而 transcript 里看不出预算是从哪儿烧掉的。
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	toolName = "delegate"
	// 默认预算。和以前内核里的数字一致，改设置就能变。
	defaultSteps = 12
	defaultMinutes = 15
	defaultDepth = 2
)

func readFrame(r *bufio.Reader) (map[string]interface{}, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := int(hdr[0])<<24 | int(hdr[1])<<16 | int(hdr[2])<<8 | int(hdr[3])
	body := make([]byte, n)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, err
	}
	var msg map[string]interface{}
	return msg, json.Unmarshal(body, &msg)
}

func writeFrame(w io.Writer, v interface{}) {
	b, _ := json.Marshal(v)
	hdr := []byte{byte(len(b) >> 24), byte(len(b) >> 16), byte(len(b) >> 8), byte(len(b))}
	w.Write(hdr)
	w.Write(b)
	if f, ok := w.(interface{ Flush() error }); ok {
		f.Flush()
	}
}

func main() {
	in := bufio.NewReader(os.Stdin)
	out := bufio.NewWriter(os.Stdout)

	for {
		msg, err := readFrame(in)
		if err != nil {
			return
		}
		switch msg["op"] {
		case "describe":
			writeFrame(out, map[string]interface{}{
				"tools": []interface{}{map[string]interface{}{
					"name": toolName,
					"description": delegateDescription(),
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"task": map[string]interface{}{
								"type": "string",
								"description": "The whole subtask, standing on its own: what to do, where, and what counts as finished.",
							},
							"model": map[string]interface{}{
								"type": "string",
								"description": "Optional model for the subtask. Defaults to the session's. A cheaper model is usually right for mechanical work.",
							},
							"packs": map[string]interface{}{
								"type": "string",
								"description": "Which packs the subtask loads: `none` (base tools only, 0.2 s), `same` (this conversation's, the default) or `all`. Pass `none` unless the task needs a tool this conversation has.",
							},
						},
						"required": []string{"task"},
					},
				}},
			})
		case "call":
			args, _ := msg["args"].(map[string]interface{})
			text, err := run(args)
			if err != nil {
				writeFrame(out, map[string]interface{}{"error": err.Error()})
			} else {
				writeFrame(out, map[string]interface{}{"result": text})
			}
		case "state", "register", "next":
			// 内核偶尔会问一句，这里不需要知道任何东西，但要回话，
			// 不然内核会等着。
			writeFrame(out, map[string]interface{}{"ok": true})
		default:
			writeFrame(out, map[string]interface{}{"error": "unsupported op"})
		}
	}
}

func delegateDescription() string {
	steps := defaultSteps
	minutes := defaultMinutes
	return fmt.Sprintf(
		"Run a self-contained subtask in a fresh ngu process with no memory of this "+
			"conversation, and get back only its answer. Use it to keep a large side "+
			"investigation out of this transcript. The subtask sees the same tools and "+
			"the same model you do, but not what was said here — so state everything it "+
			"needs, including paths and what 'done' looks like. It cannot ask you anything. "+
			"It stops after %d steps or %d minutes. It is one process: it can be killed, "+
			"it can fail on its own, and it does not touch this conversation.",
		steps, minutes)
}

// 找到能跑的 ngu。环境变量优先，因为包可能装在别处。
func nguPath() (string, error) {
	if p := os.Getenv("NGU_EXE"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	if p, err := exec.LookPath("ngu"); err == nil {
		return p, nil
	}
	// 和这个插件自己放在一起：包目录的上三级通常就是仓库根。
	exe, _ := os.Executable()
	dir := filepath.Dir(exe)
	for i := 0; i < 5; i++ {
		for _, name := range []string{"ngu.exe", "ngu"} {
			candidate := filepath.Join(dir, name)
			if _, err := os.Stat(candidate); err == nil {
				return candidate, nil
			}
			candidate = filepath.Join(dir, "target", "release", name)
			if _, err := os.Stat(candidate); err == nil {
				return candidate, nil
			}
		}
		dir = filepath.Dir(dir)
	}
	return "", fmt.Errorf("cannot find the `ngu` binary: set NGU_EXE to its path")
}

func run(args map[string]interface{}) (string, error) {
	task, _ := args["task"].(string)
	if strings.TrimSpace(task) == "" {
		return "", fmt.Errorf("missing required argument: task. Say what to do, where, and what counts as finished.")
	}

	// 深度守卫：父进程传下来的层数 + 1。
	depth := 0
	if raw := os.Getenv("NGU_DEPTH"); raw != "" {
		fmt.Sscanf(raw, "%d", &depth)
	}
	limit := defaultDepth
	if raw := os.Getenv("NGU_DEPTH_LIMIT"); raw != "" {
		fmt.Sscanf(raw, "%d", &limit)
	}
	if depth >= limit {
		return "", fmt.Errorf(
			"subagents are already %d deep; this one may not delegate further. "+
				"Do the work here instead, or raise NGU_DEPTH_LIMIT.", depth)
	}

	binary, err := nguPath()
	if err != nil {
		return "", err
	}

	// 子进程：裸启动（不带人格、不带注入），包按调用方说的来。
	packs, _ := args["packs"].(string)
	if packs == "" {
		// 默认继承：子任务应该在这个会话允许的范围内工作。
		packs = "same"
	}

	argv := []string{"--bare", "--packs", packs, "--json", "-p", task}
	if model, _ := args["model"].(string); model != "" {
		argv = append(argv, "--model", model)
	}

	cmd := exec.Command(binary, argv...)
	cmd.Env = append(os.Environ(), fmt.Sprintf("NGU_DEPTH=%d", depth+1))
	// 子代理的输出是 JSON，别让它继承父的标准输入：它会读 stdin 当 prompt
	// 如果 -p 没给，而这里给了 -p，所以安全，但显式关掉更清楚。
	cmd.Stdin = nil

	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// 墙钟上限：一个卡住的子任务要让调用方等多久，必须有个数。
	done := make(chan error, 1)
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("starting a subagent: %v", err)
	}
	go func() { done <- cmd.Wait() }()

	select {
	case err := <-done:
		if err != nil {
			return "", fmt.Errorf("the subagent failed: %v\n%s", err, truncate(stderr.String(), 800))
		}
	case <-time.After(defaultMinutes * time.Minute):
		cmd.Process.Kill()
		return "", fmt.Errorf("the subagent ran past %d minutes and was stopped. Split the task rather than making one subtask do everything.", defaultMinutes)
	}

	return readResult(stdout.String(), stderr.String())
}

// 从 `ngu --json` 的输出里取答案。
func readResult(stdout, stderr string) (string, error) {
	trimmed := strings.TrimSpace(stdout)
	if trimmed == "" {
		return "", fmt.Errorf("the subagent said nothing.\n%s", truncate(stderr, 800))
	}
	var payload struct {
		OK    bool   `json:"ok"`
		Text  string `json:"text"`
		Error string `json:"error"`
		Steps int    `json:"steps"`
	}
	if err := json.Unmarshal([]byte(trimmed), &payload); err != nil {
		// 不是 JSON：把它当文本交回去，比丢掉强。
		return truncate(trimmed, 8000), nil
	}
	if !payload.OK {
		reason := payload.Error
		if reason == "" {
			reason = "the subagent reported failure without saying why"
		}
		return "", fmt.Errorf("%s", reason)
	}
	if payload.Text == "" {
		return "(the subtask finished without saying anything)", nil
	}
	if payload.Steps > 0 {
		// 步数写进结果，因为调用方要靠它判断"结果可信度"：一个 12 步全用完
		// 的答案通常是被截断的。
		return fmt.Sprintf("%s\n\n（子代理用了 %d 步）", payload.Text, payload.Steps), nil
	}
	return payload.Text, nil
}

func truncate(s string, max int) string {
	s = strings.TrimSpace(s)
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}
