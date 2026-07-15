NVCC = nvcc
CUTLASS_DIR = cutlass
NVCCFLAGS = -O3 -arch=sm_89 -std=c++17 -lineinfo -Xptxas -dlcm=cg --expt-relaxed-constexpr -I$(CUTLASS_DIR)/include
LDFLAGS = -lcublas

BUILD_DIR = build
TARGET = $(BUILD_DIR)/gemm
OBJS = $(BUILD_DIR)/main.o $(BUILD_DIR)/kernel.o
PYTHON = python
PY_SCRIPT = main.py

.PHONY: all clean debug cublas profile python

all: $(TARGET)
	./$(TARGET)

debug: $(TARGET)
	./$(TARGET) --debug

cublas: $(TARGET)
	./$(TARGET) --cublas

profile: $(TARGET) | $(BUILD_DIR)
	sudo ncu --set full --print-details all --import-source yes -f -o $(BUILD_DIR)/ncu_report_cutlass ./$(TARGET)
	sudo ncu --set full --print-details all --import-source yes -f -o $(BUILD_DIR)/ncu_report_cublas ./$(TARGET) --cublas
	@echo "NCU reports generated in $(BUILD_DIR)"

python: $(PY_SCRIPT)
	$(PYTHON) $(PY_SCRIPT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
