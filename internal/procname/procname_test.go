package procname

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func writeComm(t *testing.T, root string, pid, content string) {
	t.Helper()
	dir := filepath.Join(root, pid)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "comm"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestFromProc(t *testing.T) {
	root := t.TempDir()
	writeComm(t, root, "4242", "llama-server\n")
	writeComm(t, root, "1337", "Xorg\n")
	writeComm(t, root, "77", "Web Content\n") // comm may contain spaces

	for pid, want := range map[int]string{4242: "llama-server", 1337: "Xorg", 77: "Web Content", 999: "", 0: "", -1: ""} {
		if got := FromProc(root, pid); got != want {
			t.Errorf("FromProc(%d) = %q, want %q", pid, got, want)
		}
	}
}

func TestFromProcUnreadable(t *testing.T) {
	root := t.TempDir()
	// A process of another user under hidepid: the directory is there, comm is not.
	if err := os.MkdirAll(filepath.Join(root, "500"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := FromProc(root, 500); got != "" {
		t.Errorf("FromProc without comm = %q, want empty", got)
	}
}

func TestParsePS(t *testing.T) {
	for in, want := range map[string]string{
		"/opt/llama.cpp/build/bin/llama-server\n":                 "llama-server",
		"/Applications/Steam.app/Contents/MacOS/steam_osx  \n":    "steam_osx",
		"/System/Library/WindowServer\n":                          "WindowServer",
		"/Applications/Foo.app/Contents/MacOS/Foo Helper (GPU)\n": "Foo Helper (GPU)",
		"launchd\n": "launchd",
		"":          "",
		"\n":        "",
	} {
		if got := parsePS([]byte(in)); got != want {
			t.Errorf("parsePS(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestNameSelf(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("no process name lookup on " + runtime.GOOS)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Base(exe)
	if runtime.GOOS == "linux" && len(want) > 15 {
		want = want[:15] // the kernel truncates comm
	}
	if got := Name(os.Getpid()); got != want {
		t.Errorf("Name(self) = %q, want %q", got, want)
	}
	if got := Name(0); got != "" {
		t.Errorf("Name(0) = %q, want empty", got)
	}
}

func TestCacheLooksUpNewPIDsOnly(t *testing.T) {
	calls := map[int]int{}
	names := map[int]string{1: "Xorg", 2: "llama-server"}
	c := NewCache(func(pid int) string {
		calls[pid]++
		return names[pid]
	})

	got := c.Names([]int{1, 2, 3, 2})
	if got[1] != "Xorg" || got[2] != "llama-server" || got[3] != "" || len(got) != 3 {
		t.Errorf("names = %v", got)
	}
	for i := 0; i < 5; i++ {
		c.Names([]int{1, 2, 3})
	}
	if calls[1] != 1 || calls[2] != 1 || calls[3] != 1 {
		t.Errorf("every PID should be looked up once while it stays, calls = %v", calls)
	}

	// PID 2 leaves; when it comes back it may be another process.
	c.Names([]int{1, 3})
	names[2] = "game"
	if got := c.Names([]int{1, 2, 3}); got[2] != "game" {
		t.Errorf("a returning PID should be looked up again, got %q", got[2])
	}
	if calls[2] != 2 || calls[1] != 1 {
		t.Errorf("calls = %v", calls)
	}

	if got := c.Names(nil); len(got) != 0 {
		t.Errorf("no PIDs should give no names, got %v", got)
	}
}

func TestCacheResultIsACopy(t *testing.T) {
	c := NewCache(func(int) string { return "x" })
	got := c.Names([]int{1})
	got[1] = "changed"
	if again := c.Names([]int{1}); again[1] != "x" {
		t.Errorf("mutating a result changed the cache: %q", again[1])
	}
}
