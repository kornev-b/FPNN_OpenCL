#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace std;
using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024

static const size_t work_group_size = 8;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;

cl_mem write_buf_weights;
cl_mem write_buf_src_conns;
cl_mem write_buf_tgt_conns;
cl_mem write_buf_end_conns;
cl_mem write_buf_end_nodes;
cl_mem write_buf_input_values;
cl_mem write_buf_output_ids;
cl_mem output_buf;

int nodesCount;
int inputsCount;
int outputsCount;
int connsCount;
int layersCount;
vector<double> weights;
vector<double> srcConns;
vector<double> tgtConns;
vector<double> endConns;
vector<double> endNodes;
vector<double> input;
double* output;
vector<double> outputIds;

scoped_aligned_ptr<double> weightsAligned;
scoped_aligned_ptr<double> srcConnsAligned;
scoped_aligned_ptr<double> tgtConnsAligned;
scoped_aligned_ptr<double> endConnsAligned;
scoped_aligned_ptr<double> endNodesAligned;
scoped_aligned_ptr<double> inputValuesAligned;
scoped_aligned_ptr<double> outputIdsAligned;
scoped_aligned_ptr<double> outputsAligned;

vector<vector<double>> validationSet;

// Function prototypes
bool init();
void cleanup();
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name);
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name);
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name);
static void device_info_string( cl_device_id device, cl_device_info param, const char* name);
static void display_device_info( cl_device_id device );

void initAligned();
bool initProblem();
bool initOpencl();
void run();

int main() {
	if(!initProblem())
		return -1;

	if(!initOpencl())
		return -1;

	run();

	// Free the resources allocated
	cleanup();

	return 0;
}

 bool initProblem()
{
	string genome_path("K:\\nn\\opencl\\altera\\fnn\\simple_genome.xml");
	ifstream infile(genome_path);
	if (!infile.is_open()) {
		printf("Failed to open genome file.");
		return false;
	}
	string line;
	getline(infile, line);
	nodesCount = atoi(line.c_str());
	getline(infile, line);
	inputsCount = atoi(line.c_str());
	getline(infile, line);
	outputsCount = atoi(line.c_str());
	getline(infile, line);
	layersCount = atoi(line.c_str());
	getline(infile, line);
	string buf; // Have a buffer string
	stringstream ssWeights(line); // Insert the string into a stream
	while (ssWeights >> buf)
		weights.push_back(strtod(buf.c_str(), 0));
	connsCount = weights.size();
	// source connections
	getline(infile, line);
	stringstream ssSrcConns(line);
	while (ssSrcConns >> buf)
		srcConns.push_back(atoi(buf.c_str()));
	// target connections
	getline(infile, line);
	stringstream ssTgtConns(line);
	while (ssTgtConns >> buf)
		tgtConns.push_back(atoi(buf.c_str()));
	// layers end connections
	getline(infile, line);
	stringstream ssEndConns(line);
	while (ssEndConns >> buf)
		endConns.push_back(atoi(buf.c_str()));
	// layers end nodes
	getline(infile, line);
	stringstream ssEndNodes(line);
	while (ssEndNodes >> buf)
		endNodes.push_back(atoi(buf.c_str()));
	// output ids
	getline(infile, line);
	stringstream ssOut(line);
	while (ssOut >> buf)
		outputIds.push_back(atoi(buf.c_str()));
	
	string validation_set("K:\\nn\\opencl\\altera\\fnn\\cross-validate.txt");
	ifstream vs_infile(validation_set);
	if (!vs_infile.is_open()) {
		printf("Failed to open validation dataset.");
		return false;
	}
	double feature;
	while(getline(vs_infile, line)) {
		stringstream ssIrisDatasetLine(line);
		vector<double> row;
		while (ssIrisDatasetLine >> feature)
		{
			//printf((line + "\n").c_str());
			row.push_back(feature);
			//printf("Add item %f from row %s \n", feature, line.c_str());
			if(ssIrisDatasetLine.peek() == ';')
			{
				ssIrisDatasetLine.ignore();
			}
		}
		validationSet.push_back(row);
	}

	if(validationSet.size() == 0)
	{
		printf("Failed to parse validation dataset.");
		return false;
	}

	//printf("First row size %llu", validationSet[0].size());

	inputValuesAligned.reset(validationSet[0].size() - 3);
	for(int i = 0; i < 4; i++)
	{
		inputValuesAligned[i] = validationSet[0][i];
	}

	initAligned();
}

void initAligned()
{
	weightsAligned.reset(connsCount);
	srcConnsAligned.reset(connsCount);
	tgtConnsAligned.reset(connsCount);
	for(int i = 0; i < connsCount; i++)
	{
		weightsAligned[i] = weights[i];
		srcConnsAligned[i] = srcConns[i];
		tgtConnsAligned[i] = tgtConns[i];
	}
	endConnsAligned.reset(layersCount);
	endNodesAligned.reset(layersCount);
	for(int i = 0; i < layersCount; i++)
	{
		endConnsAligned[i] = endConns[i];
		endNodesAligned[i] = endNodes[i];
	}
	outputsAligned.reset(outputsCount);
	for(int i = 0; i < outputsCount; i++)
	{
		outputsAligned[i] = 0;
	}
	outputIdsAligned.reset(outputIds.size());
	for(int i = 0; i < outputIds.size(); i++)
	{
		outputIdsAligned[i] = outputIds[i];
	}
}

/////// HELPER FUNCTIONS ///////

bool initOpencl() {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  printf("Trying to find an Altera platform.\n");
  // Get the OpenCL platform.
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Altera OpenCL platform.\n");
    return false;
  }

  // User-visible output - Platform information
  {
    char char_buffer[STRING_BUFFER_LEN]; 
    printf("Querying platform for info:\n");
    printf("==========================\n");
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
  }

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  // Display some device information.
  display_device_info(device);

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queue.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Create the program.
  std::string binary_file = getBoardBinaryFile("fnn", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  const char *kernel_name = "recurrent_nn";  // Kernel name, as defined in the CL file
  kernel = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create kernel");

  // weights input buffer.
  write_buf_weights = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        connsCount * sizeof(double), NULL, &status);
  checkError(status, "Failed to create buffer for weights input");

  // connection source ids input buffer.
  write_buf_src_conns = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        connsCount * sizeof(double), NULL, &status);
  checkError(status, "Failed to create buffer for connection source ids input");

  // connection target ids input buffer.
  write_buf_tgt_conns = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        connsCount * sizeof(double), NULL, &status);
  checkError(status, "Failed to create buffer for connection target ids input");

  // layer's end connections ids input buffer.
  write_buf_end_conns = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        layersCount * sizeof(double), NULL, &status);
  checkError(status, "Failed to create buffer for layer's end connections ids input");

  // layer's end nodes ids input buffer.
  write_buf_end_nodes = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        layersCount * sizeof(double), NULL, &status);
  checkError(status, "Failed to create buffer for layer's end nodes ids input");

  // input values buffer.
  write_buf_input_values = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        inputsCount * sizeof(double), NULL, &status);
  checkError(status, "Failed to create buffer for input values");

  // output ids buffer.
  write_buf_output_ids = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        outputsCount * sizeof(double), NULL, &status);
  checkError(status, "Failed to create buffer for output ids input");

  // outputs buffer.
  output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        outputsCount * sizeof(double), NULL, &status);
  checkError(status, "Failed to create buffer for output values");

  return true;
}

void run()
{
	cl_int status;

  	const double start_time = getCurrentTimestamp();

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
//    cl_event write_event[7];
    status = clEnqueueWriteBuffer(queue, write_buf_weights, CL_FALSE,
        0, connsCount * sizeof(double), weightsAligned, 0, NULL, NULL);
    checkError(status, "Failed to transfer weights");

    status = clEnqueueWriteBuffer(queue, write_buf_src_conns, CL_FALSE,
        0, connsCount * sizeof(double), srcConnsAligned, 0, NULL, NULL);
    checkError(status, "Failed to transfer source connection ids");

	status = clEnqueueWriteBuffer(queue, write_buf_tgt_conns, CL_FALSE,
        0, connsCount * sizeof(double), tgtConnsAligned, 0, NULL, NULL);
    checkError(status, "Failed to transfer target connection ids");

	status = clEnqueueWriteBuffer(queue, write_buf_end_conns, CL_FALSE,
        0, layersCount * sizeof(double), endConnsAligned, 0, NULL, NULL);
    checkError(status, "Failed to transfer target connection ids");

	status = clEnqueueWriteBuffer(queue, write_buf_end_nodes, CL_FALSE,
        0, layersCount * sizeof(double), endNodesAligned, 0, NULL, NULL);
    checkError(status, "Failed to transfer target connection ids");

	status = clEnqueueWriteBuffer(queue, write_buf_output_ids, CL_FALSE,
        0, outputsCount * sizeof(double), outputIdsAligned, 0, NULL, NULL);
    checkError(status, "Failed to transfer target connection ids");

	status = clEnqueueWriteBuffer(queue, write_buf_input_values, CL_FALSE,
        0, inputsCount * sizeof(double), inputValuesAligned, 0, NULL, NULL);
    checkError(status, "Failed to transfer target connection ids");

	// Wait for all queues to finish.
	clFinish(queue);

  // Set kernel arguments.
  unsigned argi = 0;

    // conns number
  status = clSetKernelArg(kernel, argi++, sizeof(cl_int), (void*)&connsCount);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

    // nodes number
  status = clSetKernelArg(kernel, argi++, sizeof(cl_int), (void*)&nodesCount);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  // inputs number
  status = clSetKernelArg(kernel, argi++, sizeof(cl_int), (void*)&inputsCount);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  // outpust number
  status = clSetKernelArg(kernel, argi++, sizeof(cl_int), (void*)&outputsCount);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  // layers number
  status = clSetKernelArg(kernel, argi++, sizeof(cl_int), (void*)&layersCount);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  // weights
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &write_buf_weights);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  // connection's source ids
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &write_buf_src_conns);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  // connection's target ids
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &write_buf_tgt_conns);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  // array with layer's end connection ids
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &write_buf_end_conns);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  // array with layer's end nodes ids
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &write_buf_end_nodes);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  // outputs ids
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &write_buf_output_ids);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  // input values
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &write_buf_input_values);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  // output values
  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
  checkError(status, "Failed to set kernel arg %d", argi - 1);

  printf("\nKernel initialization is complete.\n");
  printf("Launching the kernel...\n\n");

  // Configure work set over which the kernel will execute
  size_t wgSize[3] = {work_group_size, 1, 1};
  size_t gSize[3] = {work_group_size, 1, 1};

  scoped_array<cl_event> kernel_event(1);
  scoped_array<cl_event> finish_event(1);

  // Launch the kernel
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gSize, wgSize, 0, NULL, kernel_event);
  checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
  status = clEnqueueReadBuffer(queue, output_buf, CL_FALSE,
        0, outputsCount * sizeof(float), outputsAligned, 1, kernel_event, finish_event);
  checkError(status, "Failed to launch kernel");

    // Release local events.
//    clReleaseEvent(write_event[0]);
//    clReleaseEvent(write_event[1]);
//    clReleaseEvent(write_event[2]);
//    clReleaseEvent(write_event[3]);
//    clReleaseEvent(write_event[4]);
//    clReleaseEvent(write_event[5]);
//	clReleaseEvent(write_event[6]);

  // Wait for command queue to complete pending events
  clWaitForEvents(1, finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

  printf("\nKernel execution is complete.\n");
}

// Free the resources allocated during initialization
void cleanup() {
  if(kernel) {
    clReleaseKernel(kernel);  
  }
  if(program) {
    clReleaseProgram(program);
  }
  if(queue) {
    clReleaseCommandQueue(queue);
  }
  if(context) {
    clReleaseContext(context);
  }
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
   cl_ulong a;
   clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
   printf("%-40s = %llu\n", name, a);
}
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
   cl_uint a;
   clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
   printf("%-40s = %u\n", name, a);
}
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
   cl_bool a;
   clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
   printf("%-40s = %s\n", name, (a?"true":"false"));
}
static void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
   char a[STRING_BUFFER_LEN]; 
   clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
   printf("%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info( cl_device_id device ) {

   printf("Querying device for info:\n");
   printf("========================\n");
   device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
   device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
   device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
   device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
   device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
   device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
   device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
   device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
   device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
   device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
   device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
   device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
   device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

   {
      cl_command_queue_properties ccp;
      clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
      printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
      printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
   }
}