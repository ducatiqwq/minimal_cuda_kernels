NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_89 -std=c++17 --expt-relaxed-constexpr

BUILD_DIR = build
TARGET = $(BUILD_DIR)/toture_tensor_core
OBJS = $(BUILD_DIR)/main.o $(BUILD_DIR)/kernel.o
REPORT = $(BUILD_DIR)/ncu_report

.PHONY: all clean profile

all: $(TARGET)
	./$(TARGET)

profile: $(TARGET) | $(BUILD_DIR)
	sudo ncu --print-details all ./$(TARGET) > $(REPORT)
	@echo "Report saved to $(REPORT)"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
