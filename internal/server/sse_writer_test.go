package server

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestSSEWriterSoftGoneAndShortWrite(t *testing.T) {
	rec := httptest.NewRecorder()
	sw := newSSEWriter(rec, rec, context.Background())
	if err := sw.WriteString("data: hi\n\n", false); err != nil {
		t.Fatal(err)
	}
	if !sw.LastOK() || !strings.Contains(rec.Body.String(), "hi") {
		t.Fatalf("write failed: ok=%v body=%q", sw.LastOK(), rec.Body.String())
	}

	// Soft fail once via softFailRecorder
	soft := &softFailRecorder{ResponseRecorder: httptest.NewRecorder(), failWrites: 1, failAny: true}
	sw2 := newSSEWriter(soft, soft, context.Background())
	if err := sw2.WriteString("data: x\n\n", true); err != nil {
		t.Fatalf("soft write should swallow, got %v", err)
	}
	if !sw2.SoftGone() || sw2.LastOK() {
		t.Fatalf("expected softGone without lastOK, gone=%v ok=%v", sw2.SoftGone(), sw2.LastOK())
	}
}

func TestSSEWriterKeepaliveThrottle(t *testing.T) {
	rec := httptest.NewRecorder()
	sw := newSSEWriter(rec, rec, context.Background())
	if err := sw.Keepalive(": keepalive\n\n", 50*time.Millisecond, true); err != nil {
		t.Fatal(err)
	}
	if err := sw.Keepalive(": keepalive\n\n", 50*time.Millisecond, true); err != nil {
		t.Fatal(err)
	}
	// Second immediate keepalive must be throttled (body still one frame).
	if n := strings.Count(rec.Body.String(), "keepalive"); n != 1 {
		t.Fatalf("want 1 keepalive, got %d body=%q", n, rec.Body.String())
	}
	time.Sleep(60 * time.Millisecond)
	if err := sw.Keepalive(": keepalive\n\n", 50*time.Millisecond, true); err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(rec.Body.String(), "keepalive"); n != 2 {
		t.Fatalf("want 2 keepalives after interval, got %d", n)
	}
}

func TestSSEWriterCtxCancelSoft(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	rec := httptest.NewRecorder()
	sw := newSSEWriter(rec, rec, ctx)
	if err := sw.WriteString("data: a\n\n", false); err != nil {
		t.Fatalf("non-force on cancelled ctx should soft-skip, got %v", err)
	}
	if !sw.SoftGone() {
		t.Fatal("expected softGone on cancelled ctx")
	}
	// force still attempts write
	if err := sw.WriteString("data: b\n\n", true); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(rec.Body.String(), "b") {
		t.Fatalf("force write should land, body=%q", rec.Body.String())
	}
}

// Ensure softFailRecorder is usable as ResponseWriter+Flusher for sseWriter tests.
var _ interface {
	Write([]byte) (int, error)
	Flush()
} = (*softFailRecorder)(nil)
