include config.mk

.PHONY: all build run test fmt lint clean sweep install uninstall bench photons-bench render

all: build

build: sweep
	$(CARGO) build --release --locked

run: sweep
	$(CARGO) run -- $(ARGS)

test: sweep
	$(CARGO) test --workspace --all-targets
	$(CARGO) test --workspace --doc

fmt:
	$(CARGO) fmt --all

lint: sweep
	$(CARGO) clippy --workspace --all-targets --locked -- -D warnings

# Run the headless GPU benchmark gate on the active Vulkan adapter.
bench: sweep
	$(CARGO) run --release -- bench $(ARGS)

# Run the headless forward photon benchmark on the active Vulkan adapter.
photons-bench: sweep
	$(CARGO) run --release -- photons bench $(ARGS)

# Render a headless Kerr black hole PNG.
render: sweep
	$(CARGO) run --release -- render --output target/render.png $(ARGS)

clean:
	$(CARGO) clean

sweep:
	@size=$$(du -s --block-size=1G target 2>/dev/null | cut -f1); \
	if [ -n "$$size" ] && [ "$$size" -ge $(TARGET_MAX) ]; then \
		echo "target/ is $${size}G, sweeping workspace builds"; \
		$(CARGO) clean --workspace; \
		size=$$(du -s --block-size=1G target 2>/dev/null | cut -f1); \
		if [ -n "$$size" ] && [ "$$size" -ge $(TARGET_MAX) ]; then $(CARGO) clean; fi; \
	fi

install:
	install -Dm755 target/release/$(APP) $(DESTDIR)$(PREFIX)/bin/$(APP)

uninstall:
	$(RM) $(DESTDIR)$(PREFIX)/bin/$(APP)
