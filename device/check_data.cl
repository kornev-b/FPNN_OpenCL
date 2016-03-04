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