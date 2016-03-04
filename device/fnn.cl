// steepend sigmoid
#define BLOCK_SIZE 18 // depends on max(nodes,connections) number
#define GROUP_SIZE 1 // global work size that set in the host (number of work groups should be equal to 1)

float activate(float value);

__kernel
__attribute((reqd_work_group_size(1,1,1)))
 void recurrent_nn(int connsCount, int nodesNumber, int inputsCount, int outputsCount, 
	int layersCount, __global const float *restrict w, __global const int *restrict sc, __global const int *restrict tgc, 
	__global const int *restrict ec, __global const int *restrict en, __global const int *restrict oids, __global const float *restrict in, __global float *restrict outputs) {  
	int i;
	__local float aa[BLOCK_SIZE];
    // #pragma unroll 4
	for(i = 1; i < inputsCount + 1; i++) 
	{
		aa[i] = in[i - 1];
		// printf("a[%d]=%f\n", i, aa[i]);
	}
	//printf("Setting up activation function array...\n");
	// take bias into the account
	// #pragma unroll 9
	for(i = inputsCount + 1; i < nodesNumber; i++) {
		aa[i] = 0;
		// printf("a[%d]=%f\n", i, aa[i]);
	}
	// bias
	aa[0] = 1;
	// Process all layers in turn.
	// take bias into the account
	int conId=0, nodeId=inputsCount + 1;
	mem_fence(CLK_LOCAL_MEM_FENCE);
	// #pragma unroll 5
	for(int layerId = 1; layerId < layersCount; layerId++) {
		// printf("\nCalculating weighted sum for layer %d:\n", layerId);
		int endConnId = ec[layerId - 1];
		// printf("End connection for layer %d is %d.\n", layerId, endConnId);
		// Push signals through the previous layer's connections to the current layer's nodes. 
		for(; conId < endConnId; conId++) {
			// printf("New connection. Src is %d, tgt is %d, weight is %f.\n", sc[conId], tgc[conId], w[conId]);
			aa[tgc[conId]] += aa[sc[conId]] * w[conId];
		}
		mem_fence(CLK_LOCAL_MEM_FENCE);
		// printf("Activate layer %d...\n", layerId);
		// Activate current layer's nodes.
		int endNodeId = en[layerId];
		for(; nodeId < endNodeId; nodeId++) {
			// printf("Activate a[%d]=%f\n", nodeId, aa[nodeId]);
			aa[nodeId] = activate(aa[nodeId]);
			// printf("After activation: a[%d]=%f\n", nodeId, aa[nodeId]);
		}
		mem_fence(CLK_LOCAL_MEM_FENCE);
	}
	// #pragma unroll 3
	for (i = 0; i < outputsCount; ++i) 
	{
		outputs[i] = aa[oids[i]];
		printf("output[%d]=%f\t", i, outputs[i]);
	};
}

float activate(float value)
{
	return 1.0/(1.0 + exp(-4.9*value));
}