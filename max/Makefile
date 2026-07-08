NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_89 -std=c++17 --expt-relaxed-constexpr

BUILD_DIR = build
TARGET = $(BUILD_DIR)/max
OBJS = $(BUILD_DIR)/main.o $(BUILD_DIR)/kernel.o
REPORT = $(BUILD_DIR)/ncu_report
PYTHON = python
PY_SCRIPT = main.py

.PHONY: all clean debug profile python

all: $(TARGET)
	./$(TARGET)

debug: $(TARGET)
	./$(TARGET) --debug

profile: $(TARGET) | $(BUILD_DIR)
	sudo ncu ./$(TARGET) > $(REPORT)
	@echo "Report saved to $(REPORT)"

python: $(PY_SCRIPT)
	$(PYTHON) $(PY_SCRIPT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
