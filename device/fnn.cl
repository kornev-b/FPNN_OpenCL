#define ACTIVATE(value) (1.0f / (1 + exp(-value)))
#define BLOCK_SIZE 64 // default value

bool checkData(int connsCount, int nodesNumber, int inputsCount, int outputsCount, 
	int layersCount, __global const double* weights, __global const int* srcConns, __global const int* tgtConns, 
	__global const int* endConns, __global const int* endNodes, __global const int* outputsIds, __global const double* inputValues);

__kernel void recurrent_nn(int connsCount, int nodesNumber, int inputsCount, int outputsCount, 
	int layersCount, __global const double* weights, __global const int* srcConns, __global const int* tgtConns, 
	__global const int* endConns, __global const int* endNodes, __global const int* outputsIds, __global const double* inputValues, __global double *restrict outputs) {
	unsigned thread_id = get_global_id(0);
	// activation array
	__local double aa[BLOCK_SIZE];
	if(thread_id == 0) {
		if(!checkData(connsCount, nodesNumber, inputsCount, outputsCount, layersCount, weights, srcConns, tgtConns, endConns, endNodes, outputsIds, inputValues))
		{
			printf("\nError is occurred. Invalid input arguments.");
			return;
		}
		printf("Setting up activation function array...");
		for(int i = 0; i < inputsCount; i++) {
			aa[i] = inputValues[i];
			printf("a[%d]=%f", i, aa[i]);
		}
		for(int i = inputsCount; i < nodesNumber; i++) {
			aa[i] = 0;
			printf("a[%d]=%f", i, aa[i]);
		}
		//barrier(CL_LOCAL_MEM_FENCE);

		// Process all layers in turn.
		int conId=0, nodeId=inputsCount;
		for(int layerId=1; layerId < layersCount; layerId++) {
			printf("\nCalculating weighted sum for layer %d:\n", layerId);
			int endConnId = endConns[layerId];
			printf("End connection for layer %d is %d.\n", layerId, endConnId);
			// Push signals through the previous layer's connections to the current layer's nodes.
			for(; conId < endConnId; conId++) {
				printf("New connection. Src is %d, tgt is %d, weights is %f.\n", tgtConns[conId], srcConns[conId], weights[conId]);
				aa[tgtConns[conId]] += aa[srcConns[conId]] * weights[conId];
			}
			printf("Activate layer %d...\n", layerId);
			// Activate current layer's nodes.
			int endNodeId = endNodes[layerId];
			for(; nodeId < endNodeId; nodeId++) {
				printf("Activate a[%d]=%f\n", nodeId, aa[nodeId]);
				aa[nodeId] = ACTIVATE(aa[nodeId]);
				printf("After activation: a[%d]=%f\n", nodeId, aa[nodeId]);
			}
			//barrier(CL_LOCAL_MEM_FENCE);
		}
		for (int i = 0; i < outputsCount; ++i) {
    		outputs[i] = aa[outputsIds[i]];
			printf("output[%d]=%f", i, outputs[i]);
		}
    }
}

bool checkData(int connsCount, int nodesNumber, int inputsCount, int outputsCount, 
	int layersCount, __global const double* weights, __global const int* srcConns, __global const int* tgtConns, 
	__global const int* endConns, __global const int* endNodes, __global const int* outputsIds, __global const double* inputValues)
{
	printf("Nodes number = %d\n", nodesNumber);
	printf("Inputs count = %d\n", inputsCount);
	printf("Outputs count = %d\n", outputsCount);
	printf("Layers count = %d\n", layersCount);
	for(int i = 0; i < connsCount; i++) 
	{
		printf("weights[%d]=%f\t", i, weights[i]);
	} 
	printf("\n");
	for(int i = 0; i < connsCount; i++) 
	{
		if(srcConns[i] < 0 || srcConns[i] > connsCount)
			return false;
		printf("srcConns[%d]=%d\t", i, srcConns[i]);
	} 
	printf("\n");
	for(int i = 0; i < connsCount; i++) 
	{
		if(tgtConns[i] < 0 || tgtConns[i] > connsCount)
			return false;
		printf("tgtConns[%d]=%d\t", i, tgtConns[i]);
	} 
	printf("\n");
	for(int i = 0; i < layersCount; i++)
	{
		if(endConns[i] < 0 || endConns[i] > connsCount)
			return false;
		printf("endConns[%d]=%d\t", i, endConns[i]);
	}
	printf("\n");
	for(int i = 0; i < layersCount; i++)
	{
		if(endNodes[i] < 0 || endNodes[i] > nodesNumber)
			return false;
		printf("endNodes[%d]=%d\t", i, endNodes[i]);
	}
	printf("\n");
	for(int i = 0; i < outputsCount; i++)
	{
		if(outputsIds[i] < 0 || outputsIds[i] > outputsCount)
			return false;
		printf("outputsIds[%d]=%d\t", i, outputsIds[i]);
	}
	printf("\n");
	for(int i = 0; i < inputsCount; i++)
	{
		printf("inputValues[%d]=%f\t", i, inputValues[i]);
	}
	printf("\n");
	return true;
}