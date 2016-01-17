// steepend sigmoid
#define BLOCK_SIZE 64 // depends on max(nodes,connections) number
#define GROUP_SIZE 32 // global work size that set in the host (number of work groups should be equal to 1)

bool checkData(int connsCount, int nodesNumber, int inputsCount, int outputsCount, 
	int layersCount, __global const double* weights, __global const int* srcConns, __global const int* tgtConns, 
	__global const int* endConns, __global const int* endNodes, __global const int* outputsIds, __global const double* inputValues);

double activate(double value);

__kernel void recurrent_nn(int connsCount, int nodesNumber, int inputsCount, int outputsCount, 
	int layersCount, __global const double* weights, __global const int* srcConns, __global const int* tgtConns, 
	__global const int* endConns, __global const int* endNodes, __global const int* outputsIds, __global const double* inputValues, __global double *restrict outputs) {

	private unsigned global_id = get_global_id(0);
	/*__local bool data_is_correct;
	if(global_id == 0) 
	{
		data_is_correct = checkData(connsCount, nodesNumber, inputsCount, outputsCount, layersCount, weights, srcConns, tgtConns, endConns, endNodes, outputsIds, inputValues);
		if(!data_is_correct)
		{
			printf("\nError is occurred. Invalid input arguments.");
			return;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(!data_is_correct)
		return;
	*/
    private const int item_id = get_local_id(0);    
    private const int group_id = get_group_id(0);   
    private const int group_count = get_num_groups(0);  

	private int connsCountLcl = connsCount;
	private int nodesNumberLcl = nodesNumber;
	private int inputsCountLcl = inputsCount;
	private int outputsCountLcl = outputsCount;
	private int layersCountLcl = layersCount;
	__local double aa[BLOCK_SIZE];
	__local double w[BLOCK_SIZE];
	__local int srcConnsLcl[BLOCK_SIZE];
	__local int tgtConnsLcl[BLOCK_SIZE];
	__local int endConnsLcl[BLOCK_SIZE];
	__local int endNodesLcl[BLOCK_SIZE];
	__local int outIdsLcl[BLOCK_SIZE];
	__local double inputLcl[BLOCK_SIZE];

	int i;
	for(i = item_id; i < connsCountLcl; i+= GROUP_SIZE)
	{
		w[i] = weights[i];
		srcConnsLcl[i] = srcConns[i];
		tgtConnsLcl[i] = tgtConns[i];
	}
	for(i = item_id; i < layersCountLcl; i+= GROUP_SIZE)
	{
		endConnsLcl[i] = endConns[i];
		endNodesLcl[i] = endNodes[i];
	}
	for(i = item_id; i < outputsCountLcl; i+= GROUP_SIZE)
	{
		outIdsLcl[i] = outputsIds[i];
	}
	// skip bias
	for(i = item_id + 1; i < inputsCountLcl + 1; i+= GROUP_SIZE)
	{
		inputLcl[i] = inputValues[i - 1];
		aa[i] = inputLcl[i];
		printf("a[%d]=%f\n", i, aa[i]);
	}

	//printf("Setting up activation function array...\n");
	// take bias into the account
	for(i = inputsCountLcl + item_id + 1; i < nodesNumberLcl; i+= GROUP_SIZE) {
		aa[i] = 0;
		printf("a[%d]=%f\n", i, aa[i]);
	}
	// bias
	if(global_id == 0)
		aa[0] = 1;
	barrier(CLK_LOCAL_MEM_FENCE);
	// Process all layers in turn.
	// take bias into the account
	int conId=item_id, nodeId=inputsCountLcl + item_id + 1;
	if(group_id == 0)
	{
		for(int layerId = 1; layerId < layersCountLcl; layerId++) {
			//printf("\nCalculating weighted sum for layer %d:\n", layerId);
			int endConnId = endConnsLcl[layerId - 1];
			//printf("End connection for layer %d is %d.\n", layerId, endConnId);
			// Push signals through the previous layer's connections to the current layer's nodes.
			for(; conId < endConnId; conId += GROUP_SIZE) {
				printf("New connection. Src is %d, tgt is %d, weight is %f.\n", srcConnsLcl[conId], tgtConnsLcl[conId], w[conId]);
				aa[tgtConnsLcl[conId]] += aa[srcConnsLcl[conId]] * w[conId];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			//printf("Activate layer %d...\n", layerId);
			// Activate current layer's nodes.
			int endNodeId = endNodes[layerId];
			for(; nodeId < endNodeId; nodeId += GROUP_SIZE) {
				printf("Activate a[%d]=%f\n", nodeId, aa[nodeId]);
				aa[nodeId] = activate(aa[nodeId]);
				printf("After activation: a[%d]=%f\n", nodeId, aa[nodeId]);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	if(global_id == 0) 
	{
		for (i = 0; i < outputsCountLcl; ++i) 
		{
    		outputs[i] = aa[outIdsLcl[i]];
			printf("output[%d]=%f\t", i, outputs[i]);
		};
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
		if(outputsIds[i] < 0 || outputsIds[i] > nodesNumber) 
		{
			printf("invalid output id: %d, outputsCount=%d\n", outputsIds[i], outputsCount);
			return false;
		}
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

double activate(double value)
{
	return 1.0/(1.0 + exp(-4.9*value));
}