BIN      := bin/gridcore
PKG      := ./...
VERSION  ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS  := -s -w -X main.version=$(VERSION)

# Linux GPU box: override with `make deploy HOST=user@host`
HOST     ?= gpu-box
REMOTE   ?= ~/gridcore

.PHONY: all build linux test vet lint run deploy clean

all: vet test build

build:
	CGO_ENABLED=0 go build -trimpath -ldflags '$(LDFLAGS)' -o $(BIN) ./cmd/gridcore

linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags '$(LDFLAGS)' -o $(BIN)-linux-amd64 ./cmd/gridcore

test:
	go test -race -count=1 $(PKG)

vet:
	go vet $(PKG)

run: build
	$(BIN) serve --config config.example.yaml

deploy: linux
	rsync -avz $(BIN)-linux-amd64 $(HOST):$(REMOTE)/gridcore

clean:
	rm -rf bin
