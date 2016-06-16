/*******************************************************************************
 * STEP SIMULATION 2D
 *******************************************************************************
 * T:	    Initial array with temperature at each point, including borders 
 * 	    (= boundary conditions);
 * K:	    Array specifying the thermal conductivity at each point. 
 * 	    Without borders;
 * dT:	    Array specifying the increase in temperature at each step 
 * 	    for each point, without borders;
 * n_loop:  How many pixels in a row should each thread compute. 
 * 	    must be exact fraction of blockDim.x;
 */
void integrate2D(unsigned w, unsigned h, float *T, float *K, float *dT)
{
	// convolve local_T with the laplacian operator:
	//
	//     0  1  0
	//     1 -4  1
	//     0  1  0
	//
	// and save the result in local_result
	// come mai non usiamo local_T e poi salviamo tutto indietro? 
	unsigned i, j;
	for (i = 1; i < h-1 ; ++i) {
		for (j = 1; j < w-1; ++j){
			T[i*w+j] += K[(i-1)*(w-2)+(j-1)] 
						* (T[i*w+j+1] + T[i*w+j+1] + T[(i+1)*w+j] + T[(i-1)*w+j] 
							- 4*T[i*w+j])
						+ dT[(i-1)*(w-2)+(j-1)]; 
		}
	}
}
