/* Funciones auxiliares */

#include<opencv2/opencv.hpp>
#include<math.h>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

//************************************Ejercicio A***************************************
//Creamos la función exponencial.
double funcionGaussiana(int x, double sigma) {
	double producto = (pow(2, x) / pow(2, sigma));
	return x*exp(-0.5*producto);
}

//Creamos el vector máscara.
vector<double> VectorMascara(double sigma){

	//calculo la longitud de la máscara a partir de sigma.
	int longitud_mascara = 2 * round(3 * sigma) + 1;
	double suma = 0.0;
	vector<double> vector_mascara(longitud_mascara);

	//introduzco los valores, correspondientes de la función gaussiana, al vector mascara. Hay simetria
	int centro = longitud_mascara / 2;
	for (int i = 0; i <= centro; i++) {
		vector_mascara[centro - i] = funcionGaussiana(i, sigma);
		vector_mascara[centro + i] = vector_mascara[centro - i];
	}

	//hago la suma de los valores del vector para "normalizar".
	for (int i = 0; i < longitud_mascara; i++)
		suma += vector_mascara[i];

	//el vector máscara suma 1
	for (int i = 0; i < longitud_mascara; i++)
		vector_mascara[i] /= suma;
	return vector_mascara;
}

//Obtengo en un vector señal convolucionado, la convolución1D de un vector señal con un vector máscara.
vector<double> Convolucion1D(vector<double> v_imagen, vector<double> v_mascara) {
	vector<double> v_imagen_modificado;
	int mitad_mascara = (v_mascara.size() - 1) / 2;
	int suma = 0;
	//introducimos al principio del vector los valores del vector señal que no se ven afectados por la máscara.
	for (int i = 0; i < mitad_mascara; i++)
		v_imagen_modificado.push_back(v_imagen.at(i));

	//numero de cambios que hacemos en el vector. 
	int veces_iterables = v_imagen.size() - v_mascara.size() + 1;

	//añadimos los valores del vector que se modifican con la máscara
	for (int j = 0; j < veces_iterables; j++) {
		//creamos subvector del vector de entrada del tamaño de la máscara
		vector<double> subvector_imagen(v_mascara.size());
		for (int k = 0; k < v_mascara.size(); k++) {
			subvector_imagen[k] = v_imagen[j + k];
		}
		//hacemos la convolución1D del subvector con la mascara.
		for (int l = 0; l < v_mascara.size(); l++) {
			suma += subvector_imagen[l] * v_mascara[l];
		}
		//añadimos el valor obtenido al vector de salida. 
		v_imagen_modificado.push_back(suma);
		suma = 0.0;
	}
	//introducimos al final del vector los valores del vector señal que no se ven afectados por la máscara.
	for (int i = 0; i < mitad_mascara; i++)
		v_imagen_modificado.push_back(v_imagen.at(i + veces_iterables + mitad_mascara));

	return v_imagen_modificado;
}

//Creamos una matriz copia de la original con bordes inicializados a 0.
Mat bordesCero(Mat img, vector<double> v_mascara){
	//inicializo la imagen con bordes como una copia de la imagen original.
	Mat conbordes = img;
	//Creo matrices verticales inicializadas a 0's y las concateno a derecha e izquierda de la imagen con bordes.
	Mat vertical(img.rows, (v_mascara.size() - 1) / 2, img.type(), Scalar::all(0));
	hconcat(vertical, conbordes, conbordes);
	hconcat(conbordes, vertical, conbordes);

	//Creo matrices horizontales inicializadas a 0's y las concateno a derecha e izquierda de la imagen con bordes.
	Mat horizontal((v_mascara.size() - 1) / 2, conbordes.cols, img.type(), Scalar::all(0));
	vconcat(horizontal, conbordes, conbordes);
	vconcat(conbordes, horizontal, conbordes);

	return conbordes;
}

//Creamos una matriz copia de la original con bordes reflejados de la imagen.
Mat bordesReflejada(Mat img, vector<double> v_mascara) {
	//inicializo la imagen con bordes como una copia de la imagen original.
	Mat conbordes = img;
	int borde = (v_mascara.size() - 1) / 2;


	//Creo matrices verticales final e inicial inicializadas a 0's.
	Mat vertical_final(img.rows, (v_mascara.size() - 1) / 2, img.type(), Scalar::all(0));
	Mat vertical_inicial(img.rows, (v_mascara.size() - 1) / 2, img.type(), Scalar::all(0));
	//Modifico los valores de las matrices verticales haciendo el reflejo de la imagen original
	for (int i = 0; i < vertical_final.rows; i++) {
		for (int j = 0; j < vertical_final.cols; j++) {
			vertical_final.at<double>(i, j) = img.at<double>(i, img.cols - 1 - j);
		}
	}
	for (int i = 0; i < vertical_inicial.rows; i++) {
		for (int j = 0; j < vertical_inicial.cols; j++) {
			vertical_inicial.at<double>(i, j) = img.at<double>(i, borde - 1 - j);
		}
	}
	hconcat(vertical_inicial, conbordes, conbordes);
	hconcat(conbordes, vertical_final, conbordes);


	//Creo matrices horizontales final e inicial inicializadas a 0's.
	Mat horizontal_final((v_mascara.size() - 1) / 2, conbordes.cols, img.type(), Scalar::all(0));
	Mat horizontal_inicial((v_mascara.size() - 1) / 2, conbordes.cols, img.type(), Scalar::all(0));
	//Modifico los valores de las matrices horizontales haciendo el reflejo de la imagen con los bordes verticales. 
	for (int i = 0; i < horizontal_final.rows; i++) {
		for (int j = 0; j < horizontal_final.cols; j++) {
			horizontal_final.at<double>(i, j) = conbordes.at<double>(img.rows - 1 - i, j);
		}
	}
	for (int i = 0; i < horizontal_inicial.rows; i++) {
		for (int j = 0; j < horizontal_inicial.cols; j++) {
			horizontal_inicial.at<double>(i, j) = conbordes.at<double>(borde - 1 - i, j);
		}
	}
	vconcat(horizontal_inicial, conbordes, conbordes);
	vconcat(conbordes, horizontal_final, conbordes);

	return conbordes;
}

//Hacemos la convolución2D de uma imagen img de un canal con un vector máscara.
//Para ello nos ayudamos de la imagen con bordes (img_conv_aux)
Mat Gaussiana1C(Mat img, Mat img_conv_aux, vector<double> v_mascara) {
	//para cada fila de la imagen...
	for (int i = 0; i < img_conv_aux.rows; i++) {
		vector<double> fila;
		vector<double> fila_mod;
		for (int k = 0; k < img_conv_aux.cols; k++) {
			//inicializamos la fila con los valores de la fila de la imagen con bordes.
			fila.push_back(img_conv_aux.at<double>(i, k));
		}
		//obtengo el vector convolucionado de la fila de la imagen con bordes y la mascara. 
		fila_mod = Convolucion1D(fila, v_mascara);
		//modifico los valores de las filas de la imagen con bordes, por los valores obtenido en la convolución
		for (int j = 0; j < img_conv_aux.cols; j++) {
			img_conv_aux.at<double>(i, j) = fila_mod[j];
		}
	}
	//para cada columna de la imagen hago el mismo proceso que hemos hecho anteriormente para las filas. 
	for (int i = 0; i < img_conv_aux.cols; i++) {
		vector<double> columna;
		vector<double> columna_mod;
		for (int k = 0; k < img_conv_aux.rows; k++) {
			columna.push_back(img_conv_aux.at<double>(k, i));
		}
		columna_mod = Convolucion1D(columna, v_mascara);
		for (int j = 0; j < img_conv_aux.rows; j++) {
			img_conv_aux.at<double>(j, i) = columna_mod[j];
		}
	}
	int borde = (v_mascara.size() - 1) / 2;
	//los bordes de la imagen convolucionada ya no nos hacen falta...así que prescindimos de ellos
	//hacemos otra copia de la imagen y establecemos la región de la imagen convolucionada sin bordes. 
	Mat img_conv = img;
	for (int i = 0; i < img_conv.cols; i++) {
		for (int j = 0; j < img_conv.rows; j++) {
			img_conv.at<double>(j, i) = img_conv_aux.at<double>(j + borde, i + borde);
		}
	}
	return img_conv;
}

//Hacemos la convolución2D de uma imagen img de tres canales con un vector máscara.
Mat Gaussiana3C(Mat img, vector<double> v_mascara) {
	Mat img_conv_aux = img;
	Mat img_conv = img;
	vector<Mat> canales_img(img.channels());
	vector<Mat> canales_img_conv_aux(img.channels());
	vector<Mat> canales_img_conv(img.channels());
	//separamos los canales de la imagen original
	split(img, canales_img);
	//separamos los canales de la imagen con bordes
	split(img_conv_aux, canales_img_conv_aux);
	//separamos los canales de la imagen que contendrá la convolución de la imagen original.
	split(img_conv, canales_img_conv);

	//obtengo las imagenes con bordes de tres canales de la imagen.
	canales_img_conv_aux[0] = bordesReflejada(canales_img[0], v_mascara);
	canales_img_conv_aux[1] = bordesReflejada(canales_img[1], v_mascara);
	canales_img_conv_aux[2] = bordesReflejada(canales_img[2], v_mascara);

	//hago la convolución1D de cada uno de los canales
	canales_img_conv[0] = Gaussiana1C(canales_img[0], canales_img_conv_aux[0], v_mascara);
	canales_img_conv[1] = Gaussiana1C(canales_img[1], canales_img_conv_aux[1], v_mascara);
	canales_img_conv[2] = Gaussiana1C(canales_img[2], canales_img_conv_aux[2], v_mascara);

	//combino los tres canales para mostrar una sola imagen que será la de convolución.
	merge(canales_img_conv, img_conv);

	return img_conv;

}

//Hacemos el filtro Gaussiano de una imagen con un vector máscara.
Mat filtroGaussiano(Mat img, vector<double> v_mascara) {
	img.convertTo(img, CV_64F);
	//Creo la imagen con bordes.
	Mat img_conv_aux = bordesReflejada(img, v_mascara);
	//descomentar para bordes cero y comentar bordesReflejada!!!!
	//Mat img_conv_aux = bordesCero(img, v_mascara); 
	Mat img_conv = img;

	//Si es una imagen de un canal, hacemos la Gaussiana1C
	if (img.channels() == 1) {
		img_conv = Gaussiana1C(img, img_conv_aux, v_mascara);
	} //si tenemos una imagen a color, tenemos que hacer la Gaussiana3C.
	else if (img.channels() == 3) {
		img_conv = Gaussiana3C(img, v_mascara);
	}

	return img_conv;
}

//Función que calcula la convolución 2D de una imagen con una máscara.
//La máscara es extraída del muestreo de una Gaussiana 2D simétrica.
Mat filtroGaussiano(Mat img, double sigma) {
	//Calculamos el vector máscara.
	vector<double> vector_mascara = VectorMascara(sigma);
	//Hacemos el filtroGaussiano de la imagen original con el vector mascara calculado. 
	Mat img_convolucion = filtroGaussiano(img, vector_mascara);
	return img_convolucion;
}

Mat frecuenciaBaja(Mat mat, double sigma) {
	Mat aux = filtroGaussiano(mat, sigma);		 //filtro gaussiano nos da la imagen de baja frecuencia.
	return aux;
}

//Función que genera la imagen de alta frecuencia de una imagen a partir de un sigma. 
Mat frecuenciaAlta(Mat mat, double sigma) {
	Mat aux = frecuenciaBaja(mat, sigma);
	aux.convertTo(aux, CV_64F);
	mat.convertTo(mat, CV_64F);

	resize(mat, mat, Size(aux.cols, aux.rows));
	Mat alta = mat - aux;	//imagen inicial - su imagen de baja frecuencia	nos da la imagen de alta frecuencia.

	//Si tenemos valores negativos en la imagen, los ponemos a 0
	for (int i = 0; i < alta.rows; i++) {
		for (int j = 0; j < alta.cols; j++) {
			if (alta.at<double>(i, j) < 0.0)
				alta.at<double>(i, j) = 0.0;
		}
	}
	return alta;
}

