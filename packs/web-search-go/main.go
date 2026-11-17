// web-search，用 Go 重写的一个包。
//
// 原来那个包（packs/web-search）声明了 `via_proxy: mcp-proxy`，意思是把它
// 交给代理暴露。但代理只在同一个 assembly 里才被启动 —— 单独装那个包，
// 工具就永远不出现，而且报告说 skipped 而不是 failed。
//
// 这个包不去改内核、不去修那个包：它只是一个**不同的包**，直接暴露自己的
// 工具。这就是"另写一个包来改进它"的意思。
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// 一条帧：4 字节大端长度 + JSON。
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

// 三家服务商，各自一种方言。key 从 settings 的 packs 表里取。
type provider struct {
	name     string
	endpoint string
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
					"name": "search_web",
					"description": "Search the web. One tool over Tavily, Brave and Exa. " +
						"Use it when a question needs current information. " +
						"Returns the top results with title, URL and a snippet.",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"query":       map[string]interface{}{"type": "string", "description": "What to search for."},
							"max_results": map[string]interface{}{"type": "integer", "description": "How many results (default 5)."},
						},
						"required": []string{"query"},
					},
				}},
				"settings_key": "web-search",
			})
		case "call":
			args, _ := msg["args"].(map[string]interface{})
			text, err := search(args)
			if err != nil {
				writeFrame(out, map[string]interface{}{"error": err.Error()})
			} else {
				writeFrame(out, map[string]interface{}{"result": text})
			}
		default:
			writeFrame(out, map[string]interface{}{"error": "unsupported op"})
		}
	}
}

// 配置从内核递来的 settings 里取；没有 key 就说清楚缺什么。
func providerFromSettings() (provider, string, error) {
	raw := os.Getenv("NGU_PACK_SETTINGS")
	if raw == "" {
		return provider{}, "", fmt.Errorf("no settings: the kernel did not pass NGU_PACK_SETTINGS")
	}
	var all map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &all); err != nil {
		return provider{}, "", fmt.Errorf("settings are not JSON: %v", err)
	}
	entry, _ := all["web-search"].(map[string]interface{})
	if entry == nil {
		return provider{}, "", fmt.Errorf("no `web-search` settings. Set them: ngu config set --plugin web-search.provider=tavily --plugin web-search.api_key=...")
	}
	name, _ := entry["provider"].(string)
	key, _ := entry["api_key"].(string)
	if name == "" || key == "" {
		return provider{}, "", fmt.Errorf("web search is not configured: set `web-search.provider` (tavily, brave or exa) and `web-search.api_key` in the settings file. No key, no tool.")
	}
	endpoint, _ := entry["endpoint"].(string)
	return provider{name: strings.ToLower(name), endpoint: endpoint}, key, nil
}

func search(args map[string]interface{}) (string, error) {
	p, key, err := providerFromSettings()
	if err != nil {
		return "", err
	}
	query, _ := args["query"].(string)
	if strings.TrimSpace(query) == "" {
		return "", fmt.Errorf("missing required argument: query")
	}
	limit := 5
	if v, ok := args["max_results"].(float64); ok && v > 0 {
		limit = int(v)
	}

	client := &http.Client{Timeout: 20 * time.Second}
	var req *http.Request
	switch p.name {
	case "tavily":
		body, _ := json.Marshal(map[string]interface{}{
			"api_key": key, "query": query, "max_results": limit,
		})
		req, _ = http.NewRequest("POST", "https://api.tavily.com/search", strings.NewReader(string(body)))
		req.Header.Set("Content-Type", "application/json")
	case "brave":
		u := fmt.Sprintf("https://api.search.brave.com/res/v1/web/search?q=%s&count=%d",
			urlEncode(query), limit)
		req, _ = http.NewRequest("GET", u, nil)
		req.Header.Set("X-Subscription-Token", key)
	case "exa":
		body, _ := json.Marshal(map[string]interface{}{
			"query": query, "numResults": limit,
		})
		req, _ = http.NewRequest("POST", "https://api.exa.ai/search", strings.NewReader(string(body)))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+key)
	default:
		return "", fmt.Errorf("unknown provider %q; expected tavily, brave or exa", p.name)
	}
	if p.endpoint != "" {
		req.URL.Host = strings.TrimPrefix(strings.TrimPrefix(p.endpoint, "https://"), "http://")
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("reaching %s: %v", p.name, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode == 401 || resp.StatusCode == 403 {
		return "", fmt.Errorf("%s rejected the key (%d). Set `web-search.api_key` again.", p.name, resp.StatusCode)
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("%s answered HTTP %d: %s", p.name, resp.StatusCode, snippet(string(body)))
	}
	return flatten(p.name, body)
}

// 把三家的答案压成一套形状：标题 / 链接 / 摘要。
func flatten(name string, body []byte) (string, error) {
	var data map[string]interface{}
	if err := json.Unmarshal(body, &data); err != nil {
		return "", fmt.Errorf("not JSON: %s", snippet(string(body)))
	}
	var out strings.Builder
	emit := func(i int, title, link, text string) {
		fmt.Fprintf(&out, "%d. %s\n   %s\n   %s\n", i, title, link, firstLine(text, 300))
	}
	i := 0
	switch name {
	case "tavily":
		if rs, ok := data["results"].([]interface{}); ok {
			for _, r := range rs {
				m, _ := r.(map[string]interface{})
				i++
				emit(i, str(m["title"]), str(m["url"]), str(m["content"]))
			}
		}
	case "brave":
		if web, ok := data["web"].(map[string]interface{}); ok {
			if rs, ok := web["results"].([]interface{}); ok {
				for _, r := range rs {
					m, _ := r.(map[string]interface{})
					i++
					emit(i, str(m["title"]), str(m["url"]), str(m["description"]))
				}
			}
		}
	case "exa":
		if rs, ok := data["results"].([]interface{}); ok {
			for _, r := range rs {
				m, _ := r.(map[string]interface{})
				i++
				emit(i, str(m["title"]), str(m["url"]), str(m["text"]))
			}
		}
	}
	if i == 0 {
		return "", fmt.Errorf("no results for that query: %s", snippet(string(body)))
	}
	return out.String(), nil
}

func str(v interface{}) string { s, _ := v.(string); return s }
func firstLine(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > max { return s[:max] + "…" }
	return s
}
func snippet(s string) string {
	if len(s) > 300 { return s[:300] + "…" }
	return s
}
func urlEncode(s string) string {
	var b strings.Builder
	for _, c := range []byte(s) {
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~' {
			b.WriteByte(c)
		} else {
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}
