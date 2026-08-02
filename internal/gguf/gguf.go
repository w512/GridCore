// Package gguf reads GGUF file headers: metadata key/values and tensor
// descriptors. It never touches tensor data, so reading a 10 GB model costs
// a few hundred KB of I/O.
//
// Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

const magic = "GGUF"

// maxArrayKeep bounds how many array elements are decoded. Tokenizer arrays
// (hundreds of thousands of strings/scores) are skipped; per-layer arrays
// (tens to hundreds of ints) are kept.
const maxArrayKeep = 4096

// Value types in the KV section.
const (
	typeUint8   = 0
	typeInt8    = 1
	typeUint16  = 2
	typeInt16   = 3
	typeUint32  = 4
	typeInt32   = 5
	typeFloat32 = 6
	typeBool    = 7
	typeString  = 8
	typeArray   = 9
	typeUint64  = 10
	typeInt64   = 11
	typeFloat64 = 12
)

// Skipped marks an array that was too large to decode.
type Skipped struct {
	Len      uint64
	ElemType uint32
}

// TensorInfo describes one tensor.
type TensorInfo struct {
	Name   string
	Dims   []uint64
	Type   uint32 // ggml_type
	Offset uint64
}

// Elements is the number of scalars in the tensor.
func (t TensorInfo) Elements() uint64 {
	n := uint64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// Bytes is the on-disk (and in-VRAM) size of the tensor.
func (t TensorInfo) Bytes() uint64 {
	bs, ok := blockSizes[t.Type]
	if !ok {
		// Unknown quant: assume 1 byte/element, which is pessimistic for
		// any sub-8-bit type and roughly right for Q8.
		return t.Elements()
	}
	return t.Elements() / bs.elems * bs.bytes
}

// File is a parsed header.
type File struct {
	Version uint32
	KV      map[string]any // scalar: uint64|int64|float64|bool|string; array: []any or Skipped
	Tensors []TensorInfo
}

// Read parses the header of path.
func Read(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Parse(f)
}

// Parse reads a header from r. r must support seeking so large arrays can
// be skipped.
func Parse(r io.ReadSeeker) (*File, error) {
	p := &parser{r: bufio.NewReaderSize(r, 1<<20), seek: r}
	var mg [4]byte
	if _, err := io.ReadFull(p.r, mg[:]); err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if string(mg[:]) != magic {
		return nil, fmt.Errorf("gguf: bad magic %q", mg[:])
	}
	out := &File{KV: map[string]any{}}
	out.Version = p.u32()
	if out.Version < 2 || out.Version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", out.Version)
	}
	nTensors := p.u64()
	nKV := p.u64()
	if nTensors > 1<<20 || nKV > 1<<20 {
		return nil, fmt.Errorf("gguf: implausible counts tensors=%d kv=%d", nTensors, nKV)
	}
	for i := uint64(0); i < nKV && p.err == nil; i++ {
		key := p.str()
		typ := p.u32()
		out.KV[key] = p.value(typ)
	}
	for i := uint64(0); i < nTensors && p.err == nil; i++ {
		var t TensorInfo
		t.Name = p.str()
		nd := p.u32()
		if nd > 8 {
			return nil, fmt.Errorf("gguf: tensor %q has %d dims", t.Name, nd)
		}
		t.Dims = make([]uint64, nd)
		for d := range t.Dims {
			t.Dims[d] = p.u64()
		}
		t.Type = p.u32()
		t.Offset = p.u64()
		out.Tensors = append(out.Tensors, t)
	}
	if p.err != nil {
		return nil, fmt.Errorf("gguf: %w", p.err)
	}
	return out, nil
}

// Arch returns general.architecture.
func (f *File) Arch() string {
	s, _ := f.KV["general.architecture"].(string)
	return s
}

// Name returns general.name.
func (f *File) Name() string {
	s, _ := f.KV["general.name"].(string)
	return s
}

// archKey expands "block_count" to "<arch>.block_count".
func (f *File) archKey(key string) string {
	if strings.Contains(key, ".") && !strings.HasPrefix(key, "attention.") && !strings.HasPrefix(key, "ssm.") && !strings.HasPrefix(key, "rope.") {
		return key
	}
	return f.Arch() + "." + key
}

// Uint returns an integer-valued key. Floats are truncated. Keys without a
// dot (or starting with attention./ssm./rope.) are prefixed with the arch.
func (f *File) Uint(key string) (uint64, bool) {
	v, ok := f.KV[f.archKey(key)]
	if !ok {
		return 0, false
	}
	return toUint(v)
}

// Uints returns a per-layer array, broadcasting a scalar to n entries.
func (f *File) Uints(key string, n int) ([]uint64, bool) {
	v, ok := f.KV[f.archKey(key)]
	if !ok {
		return nil, false
	}
	if arr, isArr := v.([]any); isArr {
		out := make([]uint64, 0, len(arr))
		for _, e := range arr {
			u, ok := toUint(e)
			if !ok {
				return nil, false
			}
			out = append(out, u)
		}
		return out, true
	}
	u, ok := toUint(v)
	if !ok {
		return nil, false
	}
	out := make([]uint64, n)
	for i := range out {
		out[i] = u
	}
	return out, true
}

// Bools returns a bool array key.
func (f *File) Bools(key string) ([]bool, bool) {
	arr, ok := f.KV[f.archKey(key)].([]any)
	if !ok {
		return nil, false
	}
	out := make([]bool, 0, len(arr))
	for _, e := range arr {
		switch b := e.(type) {
		case bool:
			out = append(out, b)
		case uint64:
			out = append(out, b != 0)
		case int64:
			out = append(out, b != 0)
		default:
			return nil, false
		}
	}
	return out, true
}

// Tensor finds a tensor by exact name.
func (f *File) Tensor(name string) (TensorInfo, bool) {
	for _, t := range f.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return TensorInfo{}, false
}

// TotalBytes sums all tensor sizes.
func (f *File) TotalBytes() uint64 {
	var n uint64
	for _, t := range f.Tensors {
		n += t.Bytes()
	}
	return n
}

func toUint(v any) (uint64, bool) {
	switch x := v.(type) {
	case uint64:
		return x, true
	case int64:
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	case float64:
		if x < 0 || math.IsNaN(x) {
			return 0, false
		}
		return uint64(x), true
	case bool:
		if x {
			return 1, true
		}
		return 0, true
	}
	return 0, false
}

// ---- low-level parser ----

type parser struct {
	r    *bufio.Reader
	seek io.ReadSeeker
	err  error
}

func (p *parser) read(b []byte) {
	if p.err != nil {
		return
	}
	_, p.err = io.ReadFull(p.r, b)
}

func (p *parser) u8() uint8   { var b [1]byte; p.read(b[:]); return b[0] }
func (p *parser) u16() uint16 { var b [2]byte; p.read(b[:]); return binary.LittleEndian.Uint16(b[:]) }
func (p *parser) u32() uint32 { var b [4]byte; p.read(b[:]); return binary.LittleEndian.Uint32(b[:]) }
func (p *parser) u64() uint64 { var b [8]byte; p.read(b[:]); return binary.LittleEndian.Uint64(b[:]) }

func (p *parser) str() string {
	n := p.u64()
	if p.err != nil {
		return ""
	}
	if n > 1<<24 { // 16 MB; chat templates are ~10s of KB
		p.err = fmt.Errorf("string too long: %d", n)
		return ""
	}
	b := make([]byte, n)
	p.read(b)
	return string(b)
}

// skip discards n bytes, seeking when possible.
func (p *parser) skip(n uint64) {
	if p.err != nil {
		return
	}
	// Drain the bufio buffer first, then seek the underlying reader.
	buffered := uint64(p.r.Buffered())
	if n <= buffered {
		_, p.err = p.r.Discard(int(n))
		return
	}
	_, _ = p.r.Discard(int(buffered))
	n -= buffered
	if _, err := p.seek.Seek(int64(n), io.SeekCurrent); err != nil {
		p.err = err
		return
	}
	p.r.Reset(p.seek)
}

func scalarSize(typ uint32) (uint64, bool) {
	switch typ {
	case typeUint8, typeInt8, typeBool:
		return 1, true
	case typeUint16, typeInt16:
		return 2, true
	case typeUint32, typeInt32, typeFloat32:
		return 4, true
	case typeUint64, typeInt64, typeFloat64:
		return 8, true
	}
	return 0, false
}

func (p *parser) value(typ uint32) any {
	switch typ {
	case typeUint8:
		return uint64(p.u8())
	case typeInt8:
		return int64(int8(p.u8()))
	case typeUint16:
		return uint64(p.u16())
	case typeInt16:
		return int64(int16(p.u16()))
	case typeUint32:
		return uint64(p.u32())
	case typeInt32:
		return int64(int32(p.u32()))
	case typeFloat32:
		return float64(math.Float32frombits(p.u32()))
	case typeBool:
		return p.u8() != 0
	case typeString:
		return p.str()
	case typeUint64:
		return p.u64()
	case typeInt64:
		return int64(p.u64())
	case typeFloat64:
		return math.Float64frombits(p.u64())
	case typeArray:
		et := p.u32()
		n := p.u64()
		if p.err != nil {
			return nil
		}
		if n > maxArrayKeep {
			p.skipArray(et, n)
			return Skipped{Len: n, ElemType: et}
		}
		out := make([]any, 0, n)
		for i := uint64(0); i < n && p.err == nil; i++ {
			out = append(out, p.value(et))
		}
		return out
	}
	p.err = fmt.Errorf("unknown value type %d", typ)
	return nil
}

func (p *parser) skipArray(et uint32, n uint64) {
	if sz, ok := scalarSize(et); ok {
		p.skip(sz * n)
		return
	}
	// Strings (or nested arrays): must walk element by element.
	for i := uint64(0); i < n && p.err == nil; i++ {
		switch et {
		case typeString:
			l := p.u64()
			p.skip(l)
		case typeArray:
			iet := p.u32()
			in := p.u64()
			p.skipArray(iet, in)
		default:
			p.err = errors.New("cannot skip array element type")
		}
	}
}

// ---- ggml types ----

type blockSize struct{ elems, bytes uint64 }

// blockSizes maps ggml_type -> (elements per block, bytes per block).
var blockSizes = map[uint32]blockSize{
	0:  {1, 4},     // F32
	1:  {1, 2},     // F16
	2:  {32, 18},   // Q4_0
	3:  {32, 20},   // Q4_1
	6:  {32, 22},   // Q5_0
	7:  {32, 24},   // Q5_1
	8:  {32, 34},   // Q8_0
	9:  {32, 36},   // Q8_1
	10: {256, 84},  // Q2_K
	11: {256, 110}, // Q3_K
	12: {256, 144}, // Q4_K
	13: {256, 176}, // Q5_K
	14: {256, 210}, // Q6_K
	15: {256, 292}, // Q8_K
	16: {256, 66},  // IQ2_XXS
	17: {256, 74},  // IQ2_XS
	18: {256, 98},  // IQ3_XXS
	19: {256, 50},  // IQ1_S
	20: {32, 18},   // IQ4_NL
	21: {256, 110}, // IQ3_S
	22: {256, 82},  // IQ2_S
	23: {256, 136}, // IQ4_XS
	24: {1, 1},     // I8
	25: {1, 2},     // I16
	26: {1, 4},     // I32
	27: {1, 8},     // I64
	28: {1, 8},     // F64
	29: {256, 56},  // IQ1_M
	30: {1, 2},     // BF16
	34: {256, 54},  // TQ1_0
	35: {256, 66},  // TQ2_0
	39: {32, 17},   // MXFP4
}

// TypeName returns a readable ggml type name.
func TypeName(t uint32) string {
	names := map[uint32]string{0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
		10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
		19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
		28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}
	if n, ok := names[t]; ok {
		return n
	}
	return fmt.Sprintf("type%d", t)
}

// CacheTypeBytes returns bytes per element for a llama.cpp KV cache type
// name (-ctk/-ctv), or 0 if unknown.
func CacheTypeBytes(name string) float64 {
	switch strings.ToLower(name) {
	case "f32":
		return 4
	case "f16", "bf16":
		return 2
	case "q8_0":
		return 34.0 / 32
	case "q5_1":
		return 24.0 / 32
	case "q5_0":
		return 22.0 / 32
	case "q4_1":
		return 20.0 / 32
	case "q4_0", "iq4_nl":
		return 18.0 / 32
	}
	return 0
}
