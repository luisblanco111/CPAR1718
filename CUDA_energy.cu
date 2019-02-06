/*
 * Simulacion simplificada de bombardeo de particulas de alta energia
 *
 * Computacion Paralela (Grado en Informatica)
 * 2017/2018
 *
 * (c) 2018 Arturo Gonzalez Escribano
 *
 * Modificaciones por:
 * Luis Blanco de la Cruz
 * Rubén González Ruiz
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<cputils.h>

#define PI	3.14159f
#define UMBRAL	0.001f

/* Estructura para almacenar los datos de una tormenta de particulas */
typedef struct {
	int size;
	int *posval;
} Storm;



__global__ void inicializa (float *layerDevice){

	int id = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;	
	
	layerDevice[id] = 0.0f;	
}
/* ESTA FUNCION PUEDE SER MODIFICADA */
/* Funcion para actualizar una posicion de la capa */
__global__ void actualiza( float *layerDevice,int pos, float energia ) {
	/* 1. Calcular valor absoluto de la distancia entre el
		punto de impacto y el punto k de la capa */
	int id = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
	int distancia = pos - id;
	if ( distancia < 0 ) distancia = - distancia;

	/* 2. El punto de impacto tiene distancia 1 */
	distancia = distancia + 1;

	/* 3. Raiz cuadrada de la distancia */
	//float atenuacion = (float)distancia*distancia;
	//float atenuacion = (float)distancia / PI;
	float atenuacion = sqrtf( (float)distancia );

	/* 4. Calcular energia atenuada */
	float energia_k = energia / atenuacion;

	/* 5. No sumar si el valor absoluto es menor que umbral */
	if ( energia_k >= UMBRAL || energia_k <= -UMBRAL ){
		layerDevice[id] = layerDevice[id] + energia_k;
	}
}


__global__ void copia(float *layerDevice, float *layer_copyDevice){

	int id = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

	layer_copyDevice[id] = layerDevice[id];
}




__global__ void relaja(float *layerDevice,float *layer_copyDevice, int layer_size){

	int id = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

	/* 4.2.2. Actualizar capa, menos los extremos, usando valores del array auxiliar */
	if(id!=0 && id!=layer_size-1)
	layerDevice[id] = ( layer_copyDevice[id-1] + layer_copyDevice[id] + layer_copyDevice[id+1] ) / 3;
}


__global__ void maximosLocales(float *layerDevice, float *maximos,int *posiciones, int layer_size){

	int id = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
	
	/* 4.3. Localizar maximo */
	if(id>layer_size) return;
	if(id!=0 && id!=layer_size-1){
		/* Comprobar solo maximos locales */
		if ( layerDevice[id] > layerDevice[id-1] && layerDevice[id] > layerDevice[id+1] ) {
			maximos[id] = layerDevice[id];
			posiciones[id] = id;	
		}
		else{
			maximos[id]=-1.0f;
			posiciones[id]=-1;
		}
	}
	else /*if(id==0 || id==layer_size-1)*/{
		maximos[id]=-1.0f;
		posiciones[id]=-1;
	}

}




__global__ void reduceML(float *maxin, int *posin, float *maxout,int *posout, int vSize){
	int mitad = vSize/2;
	int id = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

	if (id>mitad) return;
	
	if(maxin[id]<maxin[id+mitad]){
    		maxout[id] = maxin[id+mitad];
		posout[id] = posin[id+mitad];
	}
	else if(maxin[id]==maxin[id+mitad]){
		if(posin[id]>posin[id+mitad]) //si la posicion id+mitad es menor, traerla hacia delante
			posout[id]=posin[id+mitad];
	}
	else{ //maxin[id]>maxin[id+mitad] dejar como esta
	}

    	// Extra element
    	if ( vSize%2 != 0 && id == 0 ){
		if(maxin[0]<maxin[vSize-1]){
    			maxout[0] = maxin[vSize-1];
			posout[0] = posin[vSize-1];
		}
		else if(maxin[0]==maxin[vSize-1]){
			if(posin[0]>posin[vSize-1]) //si la posicion id+mitad es menor, traerla hacia delante
				posout[0]=posin[vSize-1];
		}
		else{ //maxin[id]>maxin[id+mitad] dejar como esta
		}
	}
}

/* FUNCIONES AUXILIARES: No se utilizan dentro de la medida de tiempo, dejar como estan */
/* Funcion de DEBUG: Imprimir el estado de la capa */
void debug_print(int layer_size, float *layer, int *posiciones, float *maximos, int num_storms ) {
	int i,k;
	if ( layer_size <= 35 ) {
		/* Recorrer capa */
		for( k=0; k<layer_size; k++ ) {
			/* Escribir valor del punto */
			printf("%10.4f |", layer[k] );

			/* Calcular el numero de caracteres normalizado con el maximo a 60 */
			int ticks = (int)( 60 * layer[k] / maximos[num_storms-1] );

			/* Escribir todos los caracteres menos el ultimo */
			for (i=0; i<ticks-1; i++ ) printf("o");

			/* Para maximos locales escribir ultimo caracter especial */
			if ( k>0 && k<layer_size-1 && layer[k] > layer[k-1] && layer[k] > layer[k+1] )
				printf("x");
			else
				printf("o");

			/* Si el punto es uno de los maximos especiales, annadir marca */
			for (i=0; i<num_storms; i++) 
				if ( posiciones[i] == k ) printf(" M%d", i );

			/* Fin de linea */
			printf("\n");
		}
	}
}

/*
 * Funcion: Lectura de fichero con datos de tormenta de particulas
 */
Storm read_storm_file( char *fname ) {
	FILE *fstorm = cp_abrir_fichero( fname );
	if ( fstorm == NULL ) {
		fprintf(stderr,"Error: Opening storm file %s\n", fname );
		exit( EXIT_FAILURE );
	}

	Storm storm;	
	int ok = fscanf(fstorm, "%d", &(storm.size) );
	if ( ok != 1 ) {
		fprintf(stderr,"Error: Reading size of storm file %s\n", fname );
		exit( EXIT_FAILURE );
	}

	storm.posval = (int *)malloc( sizeof(int) * storm.size * 2 );
	if ( storm.posval == NULL ) {
		fprintf(stderr,"Error: Allocating memory for storm file %s, with size %d\n", fname, storm.size );
		exit( EXIT_FAILURE );
	}
	
	int elem;
	for ( elem=0; elem<storm.size; elem++ ) {
		ok = fscanf(fstorm, "%d %d\n", 
					&(storm.posval[elem*2]),
					&(storm.posval[elem*2+1]) );
		if ( ok != 2 ) {
			fprintf(stderr,"Error: Reading element %d in storm file %s\n", elem, fname );
			exit( EXIT_FAILURE );
		}
	}
	fclose( fstorm );

	return storm;
}

/*
 * PROGRAMA PRINCIPAL
 */
int main(int argc, char *argv[]) {
	int i,j,k;

	/* 1.1. Leer argumentos */
	if (argc<3) {
		fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
		exit( EXIT_FAILURE );
	}

	int layer_size = atoi( argv[1] );
	int num_storms = argc-2;
	Storm storms[ num_storms ];

	/* 1.2. Leer datos de storms */
	for( i=2; i<argc; i++ ) 
		storms[i-2] = read_storm_file( argv[i] );

	/* 1.3. Inicializar maximos a cero */
	float maximos[ num_storms ];
	int posiciones[ num_storms ];
	for (i=0; i<num_storms; i++) {
		maximos[i] = 0.0f;
		posiciones[i] = 0;
	}

	/* 2. Inicia medida de tiempo */
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	double ttotal = cp_Wtime();

	/* COMIENZO: No optimizar/paralelizar el main por encima de este punto */

	/*Calculo del tamaño de bloque y de grid*/
	int tamBlock = 256;
	int tamGrid;
	
	if (layer_size % tamBlock == 0) {
		tamGrid = layer_size/tamBlock;
	} else {
		tamGrid = (layer_size/tamBlock)+1;
	}

	/*Creacion y reserva de memoria de la matrices en Device*/

	float *layerDevice;
	float *layer_copyDevice;
	float *maxDevice;
	int *posDevice;

	cudaMalloc((void**) &layerDevice, sizeof(float) * layer_size);
	cudaMalloc((void**) &layer_copyDevice, sizeof(float) * layer_size);
	cudaMalloc((void**) &maxDevice, sizeof(float) * layer_size);
	cudaMalloc((void**) &posDevice, sizeof(int) * layer_size);
	
	float *maxAbs=(float *)malloc( sizeof(float));
	int *posAbs=(int *)malloc( sizeof(int));

	inicializa<<<tamGrid, tamBlock>>>(layerDevice);
	/* 4. Fase de bombardeos */
	for( i=0; i<num_storms; i++) {
	/*Pasar de Host a Device */	
	

		/* 4.1. Suma energia de impactos */
		/* Para cada particula */
		for( j=0; j<storms[i].size; j++ ) {
			/* Energia de impacto (en milesimas) */
			float energia = (float)storms[i].posval[j*2+1] / 1000;
			/* Posicion de impacto */
			int posicion = storms[i].posval[j*2];

			/* Para cada posicion de la capa */
				/* Actualizar posicion */
			actualiza<<<tamGrid,tamBlock>>>(layerDevice,posicion, energia);
		}

		copia<<<tamGrid, tamBlock>>>(layerDevice, layer_copyDevice);
		relaja<<<tamGrid,tamBlock>>>(layerDevice,layer_copyDevice,layer_size);
		
		maximosLocales<<<tamGrid,tamBlock>>>(layerDevice, maxDevice, posDevice, layer_size);

		for( k=layer_size; k>1; k/=2 )	
		reduceML<<<tamGrid,tamBlock>>>(maxDevice,posDevice,maxDevice,posDevice,k);
	
		cudaMemcpy(maxAbs,&maxDevice[0], sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(posAbs,&posDevice[0], sizeof(int),cudaMemcpyDeviceToHost);
	
		maximos[i]=maxAbs[0];
		posiciones[i]=posAbs[0];
	
	}


	/* FINAL: No optimizar/paralelizar por debajo de este punto */

	/* 6. Final de medida de tiempo */
	cudaDeviceSynchronize();
	ttotal = cp_Wtime() - ttotal;

	/* 7. DEBUG: Dibujar resultado (Solo para capas con hasta 35 puntos) */
	#ifdef DEBUG
	debug_print( layer_size, layer, posiciones, maximos, num_storms );
	#endif

	/* 8. Salida de resultados para tablon */
	printf("\n");
	/* 8.1. Tiempo total de la computacion */
	printf("Time: %lf\n", ttotal );
	/* 8.2. Escribir los maximos */
	printf("Result:");
	for (i=0; i<num_storms; i++)
		printf(" %d %f", posiciones[i], maximos[i] );
	printf("\n");

	/* 9. Liberar recursos */	
	for( i=0; i<argc-2; i++ )
		free( storms[i].posval );

	/* 10. Final correcto */
	return 0;
}
