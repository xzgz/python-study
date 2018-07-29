################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../test_blob.cpp 

OBJS += \
./test_blob.o 

CPP_DEPS += \
./test_blob.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/xzgz/Desktop/Ubuntu/CUDA8.0_Toolkit/include -I/media/xzgz/Ubuntu/Ubuntu/Caffe/SourceCode/build_eclipse/install/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


