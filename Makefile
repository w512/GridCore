BIN      := bin/gridcore
PKG      := ./...
VERSION  ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS  := -s -w -X main.version=$(VERSION)

# Linux GPU box: override with `make deploy HOST=user@host`
HOST     ?= gpu-box
REMOTE   ?= ~/gridcore

.PHONY: all build linux test vet lint run deploy install-service clean

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
	$(BIN) serve --config examples/fake-demo.yaml --state-dir /tmp/gridcore-state

deploy: linux
	rsync -az $(BIN)-linux-amd64 $(HOST):.local/bin/gridcore

# Install and (re)start the user service on the GPU box.
install-service: deploy
	rsync -az deploy/gridcore.service $(HOST):.config/systemd/user/gridcore.service
	ssh $(HOST) 'systemctl --user daemon-reload && systemctl --user enable gridcore >/dev/null && systemctl --user restart gridcore && sleep 1 && systemctl --user --no-pager status gridcore | head -5'

clean:
	rm -rf bin
