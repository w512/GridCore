// Package gguftest builds minimal GGUF headers for tests.
package gguftest

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Value type ids from the GGUF spec.
const (
	TypeUint32  = 4
	TypeFloat32 = 6
	TypeBool    = 7
	TypeString  = 8
	TypeArray   = 9
	TypeInt32   = 5
)

// Tensor type ids (ggml_type) used in tests.
const (
	F32  = 0
	F16  = 1
	Q4_0 = 2
	Q8_0 = 8
	Q4_K = 12
	Q6_K = 14
)

// Writer accumulates KV pairs and tensor descriptors.
type Writer struct {
	buf     bytes.Buffer
	kv      bytes.Buffer
	nKV     uint64
	tensors bytes.Buffer
	nT      uint64
}

func (w *Writer) u32(b *bytes.Buffer, v uint32) { _ = binary.Write(b, binary.LittleEndian, v) }
func (w *Writer) u64(b *bytes.Buffer, v uint64) { _ = binary.Write(b, binary.LittleEndian, v) }
func (w *Writer) str(b *bytes.Buffer, s string) { w.u64(b, uint64(len(s))); b.WriteString(s) }

func (w *Writer) key(k string, typ uint32) { w.str(&w.kv, k); w.u32(&w.kv, typ); w.nKV++ }

// String adds a string value.
func (w *Writer) String(k, v string) { w.key(k, TypeString); w.str(&w.kv, v) }

// U32 adds a uint32 value.
func (w *Writer) U32(k string, v uint32) { w.key(k, TypeUint32); w.u32(&w.kv, v) }

// F32 adds a float32 value.
func (w *Writer) F32(k string, v float32) { w.key(k, TypeFloat32); w.u32(&w.kv, math.Float32bits(v)) }

// Bool adds a bool value.
func (w *Writer) Bool(k string, v bool) {
	w.key(k, TypeBool)
	if v {
		w.kv.WriteByte(1)
	} else {
		w.kv.WriteByte(0)
	}
}

// I32Array adds an int32 array.
func (w *Writer) I32Array(k string, vs []int32) {
	w.key(k, TypeArray)
	w.u32(&w.kv, TypeInt32)
	w.u64(&w.kv, uint64(len(vs)))
	for _, v := range vs {
		w.u32(&w.kv, uint32(v))
	}
}

// BoolArray adds a bool array.
func (w *Writer) BoolArray(k string, vs []bool) {
	w.key(k, TypeArray)
	w.u32(&w.kv, TypeBool)
	w.u64(&w.kv, uint64(len(vs)))
	for _, v := range vs {
		if v {
			w.kv.WriteByte(1)
		} else {
			w.kv.WriteByte(0)
		}
	}
}

// StringArray adds a string array (large ones exercise the skip path).
func (w *Writer) StringArray(k string, vs []string) {
	w.key(k, TypeArray)
	w.u32(&w.kv, TypeString)
	w.u64(&w.kv, uint64(len(vs)))
	for _, v := range vs {
		w.str(&w.kv, v)
	}
}

// F32Array adds a float array.
func (w *Writer) F32Array(k string, vs []float32) {
	w.key(k, TypeArray)
	w.u32(&w.kv, TypeFloat32)
	w.u64(&w.kv, uint64(len(vs)))
	for _, v := range vs {
		w.u32(&w.kv, math.Float32bits(v))
	}
}

// Tensor adds a tensor descriptor.
func (w *Writer) Tensor(name string, typ uint32, dims ...uint64) {
	w.str(&w.tensors, name)
	w.u32(&w.tensors, uint32(len(dims)))
	for _, d := range dims {
		w.u64(&w.tensors, d)
	}
	w.u32(&w.tensors, typ)
	w.u64(&w.tensors, 0)
	w.nT++
}

// Bytes serialises the header (no tensor data).
func (w *Writer) Bytes() []byte {
	var out bytes.Buffer
	out.WriteString("GGUF")
	w.u32(&out, 3)
	w.u64(&out, w.nT)
	w.u64(&out, w.nKV)
	out.Write(w.kv.Bytes())
	out.Write(w.tensors.Bytes())
	return out.Bytes()
}

// WriteFile writes the header to dir/name and returns the path.
func (w *Writer) WriteFile(t *testing.T, dir, name string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, w.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}
