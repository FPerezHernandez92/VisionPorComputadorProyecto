//**********************************************
//**************** Proyecto. *******************
//************ Detección facial.****************
//******** Cristina Zuheros Montes.*************
//******* Francisco Perez Hernandez. ***********
//*************** 2015-2016 ********************
//**********************************************

#include<opencv2/opencv.hpp>
#include<math.h>
#include <fstream>
#include "Header.h"

using namespace std;
using namespace cv;

/*************************************************/
/*************** Funciones auxiliares ************/
/*************************************************/

/* Función para leer imágenes */
vector<Mat> LeerImagenes(int numero_imagenes, string nombre_imagenes, int flag_color){
	vector<Mat> imagenes;
	for (int i = 1; i <= numero_imagenes; i++){
		string aux_nombre = nombre_imagenes + to_string(i) + ".jpg";
		Mat aux_imagen = imread(aux_nombre, flag_color);
		imagenes.push_back(aux_imagen);
		if (i >= 9){
			nombre_imagenes = "imagenes/image_00";
		}
		if (i >= 99){
			nombre_imagenes = "imagenes/image_0";
		}
	}
	return imagenes;
}

/* Función para pintar imágenes */
void PintaImagenes(vector<Mat> imagenes_caras, string nombre_imagenes="Salida"){
	for (int i = 0; i < imagenes_caras.size(); i++){
		string aux_nombre = nombre_imagenes + to_string(i);
		namedWindow(aux_nombre, imagenes_caras[i].channels());
		imshow(aux_nombre, imagenes_caras[i]);
		cvWaitKey();
		destroyWindow(aux_nombre);
	}
}

/* Función para pintar una imagen */
void pintaI(Mat im, char ventana[]) {
	namedWindow(ventana, 1);
	im.convertTo(im, CV_8UC3);
	imshow(ventana, im);
	cvWaitKey(0);
	destroyWindow(ventana);
}

/*************************************************/
/*************** 2. Sacar color carne ************/
/*************************************************/

/* Función para buscar el tono definido como color carne
y pasarlo a blanco en caso de coincidir o negro en caso
de no coincidir con el tono dado */
Mat PasarANegro(Mat imagen, int tolerancia=70){
	Mat salida;
	imagen.copyTo(salida);
	Vec3b pixel;
	Point punto;

	//Valores definidos para el tono color carne
	int r, g, b;
	r = 253; g = 221; b = 201;

	//Recorremos cada pixel de la imagen
	for (int i = 0; i < salida.cols; i++){
		for (int j = 0; j < salida.rows; j++){
			punto.x = i; 
			punto.y = j;
			pixel = imagen.at<Vec3b>(punto);
			if ((((int)(pixel.val[0] >(r - tolerancia))) && ((int)(pixel.val[0] < (r + tolerancia)))) &&
				(((int)(pixel.val[1] > (g - tolerancia))) && ((int)(pixel.val[1] < (g + tolerancia)))) &&
				(((int)(pixel.val[2] > (b - tolerancia))) && ((int)(pixel.val[2] < (b + tolerancia))))
				){
				//Si coincide lo pasaremos a blanco
				Vec3b aux_pixel;
				aux_pixel.val[0] = 255;
				aux_pixel.val[1] = 255;
				aux_pixel.val[2] = 255;
				salida.at<Vec3b>(punto) = aux_pixel;
			}
			else {
				//Si no coincide lo pasaremos a negro
				Vec3b aux_pixel;
				aux_pixel.val[0] = 0;
				aux_pixel.val[1] = 0;
				aux_pixel.val[2] = 0;
				salida.at<Vec3b>(punto) = aux_pixel;
			}
		}
	}
	return salida;
}

/* Función que coge una imagen de un vector dado y la transforma
a blanco-negro según la función definida anteriormente */
vector<Mat> DetectarRosa(vector<Mat> imagenes, int tolerancia){
	vector<Mat> imagenes_salida;
	for (int i = 0; i < imagenes.size(); i++){
		Mat salida;
		salida = PasarANegro(imagenes[i], tolerancia);
		imagenes_salida.push_back(salida);
	}

	//Pintamos las imágenes que vamos a usar para las pruebas
	PintaImagenes(imagenes_salida);

	return imagenes_salida;
}

/*************************************************/
/******* 3. Sacar piel de las imágenes ***********/
/*************************************************/

//Pasamos de RGB a YCrCb
Mat RGBtoYCrCb(Mat im_original){
	Mat im_conver;
	im_original.copyTo(im_conver);
	Vec3b vectorYCrCb = Vec3b(0, 0, 0);
	Vec3b vectorRGB = Vec3b(0, 0, 0);
	Mat matriztra = Mat(3, 3, CV_64FC1);
	//cambiar valores 
	//inicializo la matriz de conversión.
	matriztra.at<double>(0, 0) = 0.3;
	matriztra.at<double>(1, 0) = 0.6;
	matriztra.at<double>(2, 0) = 0.1;

	matriztra.at<double>(0, 1) = -0.2;
	matriztra.at<double>(1, 1) = -0.3;
	matriztra.at<double>(2, 1) = 0.5;

	matriztra.at<double>(0, 2) = 0.5;
	matriztra.at<double>(1, 2) = -0.4;
	matriztra.at<double>(2, 2) = -0.1;

	for (int i = 0; i < im_original.rows; i++){
		for (int j = 0; j < im_original.cols; j++)	{
			vectorRGB = Vec3b(im_original.at<Vec3b>(i, j)[0], im_original.at<Vec3b>(i, j)[1], im_original.at<Vec3b>(i, j)[2]);

			vectorYCrCb = Vec3b(vectorRGB[0] * matriztra.at<double>(0, 0) + vectorRGB[1] * matriztra.at<double>(1, 0) + vectorRGB[2] * matriztra.at<double>(2, 0),
				128 + vectorRGB[0] * matriztra.at<double>(0, 2) + vectorRGB[1] * matriztra.at<double>(1, 2) + vectorRGB[2] * matriztra.at<double>(2, 2),
				128 + vectorRGB[0] * matriztra.at<double>(0, 1) + vectorRGB[1] * matriztra.at<double>(1, 1) + vectorRGB[2] * matriztra.at<double>(2, 1));
			vectorRGB = vectorYCrCb;
			im_conver.at<Vec3b>(i, j) = vectorRGB;
		}
	}

	return im_conver;
}

//transformo de RGB a YCrCb y pinto la piel en b/n
Mat TransformarDeRGBaYCrCBYPasoABlancoNegro(Mat im, int valY, int minCr, int maxCr, int minCb, int maxCb){
	//Paso de RGB a YCrCb
	Mat salida = RGBtoYCrCb(im);

	//Pinto en blanco los niveles de piel y en negro no piel.
	for (int i = 0; i < salida.rows; i++){
		for (int j = 0; j < salida.cols; j++)	{
			if (salida.channels() == 3){
				if (salida.at<Vec3b>(i, j)[0] > valY && //Y 
					salida.at<Vec3b>(i, j)[1] > minCr && salida.at<Vec3b>(i, j)[1] < maxCr &&  //Cr
					salida.at<Vec3b>(i, j)[2] > minCb && salida.at<Vec3b>(i, j)[2] < maxCb)   //Cb
					salida.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				else
					salida.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
		}
	}

	return salida;
}

//Buscamos si hay cara y si hay pocos pixeles blancos, no identificaremos cara
bool BuscarSiHayCara(Mat im){
	bool hay = true;
	int n_blancos = 0;
	for (int i = 0; i < im.rows; i++){
		for (int j = 0; j < im.cols; j++)	{
			if (im.at<Vec3b>(i, j) == Vec3b(255, 255, 255))
				n_blancos++;
		}
	}
	if (n_blancos < (im.rows*im.cols)*0.1)
		hay = false;
	return hay;
}

Mat recortoCaraFILAS(Mat im, int &primerof, int &ultimof){

	//Obtengo un vector con el número de pixeles blancos de una fila
	vector<int> v_pixeles;
	int contador = 0;
	for (int i = 0; i < im.rows; i++){
		for (int j = 0; j < im.cols; j++)	{
			if (im.at<Vec3b>(i, j) == Vec3b(255, 255, 255))
				contador++;
		}
		v_pixeles.push_back(contador);
		contador = 0;
	}

	//Busco el máximo del vector:
	int maximo = 0;
	for (int i = 0; i < v_pixeles.size(); i++){
		if (maximo < v_pixeles.at(i))
			maximo = v_pixeles.at(i);
	}

	//Busco region de cara:
	int tolerancia = 0;
	int limite = maximo*0.3;
	int primero = 0;
	int numero_buenos = 0;
	int maximo_buenos = 0;
	int primero_bueno = 0;

	for (int i = 0; i < v_pixeles.size(); i++){
		if (tolerancia > 30){
			primero = 0;
			tolerancia = 0;
			numero_buenos = 0;
		}
		else if ((primero == 0) && (v_pixeles.at(i) >= limite)){
			primero = i;
		}
		else if ((primero != 0) && (v_pixeles.at(i) >= limite) && tolerancia <= 30){
			numero_buenos++;
			if (maximo_buenos < numero_buenos){
				maximo_buenos = numero_buenos;
				primero_bueno = primero;
			}
		}
		else if ((primero != 0) && (v_pixeles.at(i) < limite)){
			tolerancia++;
		}
	}

	//ya tengo la región de filas, ahora recorto en la imagen 
	Mat aux_filas = Mat(maximo_buenos, im.cols, CV_8UC3);
	for (int i = primero_bueno; i < primero_bueno + maximo_buenos; i++){
		for (int j = 0; j < im.cols; j++)	{
			aux_filas.at<Vec3b>(i - primero_bueno, j) = im.at<Vec3b>(i, j);
		}
	}
	primerof = primero_bueno;
	ultimof = primero_bueno + maximo_buenos;
	return aux_filas;
}

Mat recortoCaraCOL(Mat im, int &primeroc, int &ultimoc){

	//Obtengo un vector con el número de pixeles blancos de una columna
	vector<int> v_pixeles;
	int contador = 0;
	for (int i = 0; i < im.cols; i++){
		for (int j = 0; j < im.rows; j++)	{
			if (im.at<Vec3b>(j, i) == Vec3b(255, 255, 255))
				contador++;
		}
		v_pixeles.push_back(contador);
		contador = 0;
	}

	//Busco el máximo del vector:
	int maximo = 0;
	for (int i = 0; i < v_pixeles.size(); i++){
		if (maximo < v_pixeles.at(i))
			maximo = v_pixeles.at(i);
	}

	//Busco region de cara:
	int tolerancia = 0;
	int limite = maximo*0.4;
	int primero = 0;
	int numero_buenos = 0;
	int maximo_buenos = 0;
	int primero_bueno = 0;

	for (int i = 0; i < v_pixeles.size(); i++){
		if (tolerancia > 50){
			primero = 0;
			tolerancia = 0;
			numero_buenos = 0;
		}
		else if ((primero == 0) && (v_pixeles.at(i) >= limite)){
			primero = i;
		}
		else if ((primero != 0) && (v_pixeles.at(i) >= limite) && tolerancia <= 50){
			numero_buenos++;
			if (maximo_buenos < numero_buenos){
				maximo_buenos = numero_buenos;
				primero_bueno = primero;
			}
		}
		else if ((primero != 0) && (v_pixeles.at(i) < limite)){
			tolerancia++;
		}
	}

	//ya tengo la región de columnas, ahora recorto en la imagen 
	Mat aux_filas = Mat(im.rows, maximo_buenos, CV_8UC3);
	for (int i = 0; i < im.rows; i++){
		for (int j = primero_bueno; j < primero_bueno + maximo_buenos; j++)	{
			aux_filas.at<Vec3b>(i, j - primero_bueno) = im.at<Vec3b>(i, j);
		}
	}
	primeroc = primero_bueno;
	ultimoc = primero_bueno + maximo_buenos;
	return aux_filas;
}



Mat RecortarImagen(Mat im, int primero_filas, int ultimo_filas, int primero_col, int ultimo_col){
	Mat aux1 = Mat(ultimo_filas - primero_filas, im.cols, CV_8UC3);
	for (int i = primero_filas; i < ultimo_filas; i++){
		for (int j = 0; j < im.cols; j++)	{
			aux1.at<Vec3b>(i - primero_filas, j) = im.at<Vec3b>(i, j);
		}
	}

	Mat aux2 = Mat(aux1.rows, ultimo_col - primero_col, CV_8UC3);
	for (int i = 0; i < aux1.rows; i++){
		for (int j = primero_col; j < ultimo_col; j++)	{
			aux2.at<Vec3b>(i, j - primero_col) = aux1.at<Vec3b>(i, j);
		}
	}

	return aux2;
}

//Función auxiliar para buscar los ojos
vector<int> BuscarOjosProfundo(vector<int> fila, Mat imagen){
	vector<int> res;
	int columnas = imagen.cols;
	int posojo = columnas / 4;
	int seguidos = 8;
	int ojo1 = posojo - seguidos;
	int ojo2 = (posojo * 3) - seguidos;
	bool mala;
	for (int i = 0; i < fila.size(); i++){
		mala = false;
		for (int j = 0; j < seguidos * 2 && !mala; j++){
			if ((imagen.at<Vec3b>(fila[i], ojo1+j) != Vec3b(0, 0, 0)) && (imagen.at<Vec3b>(fila[i], ojo2+j) != Vec3b(0, 0, 0))){
				mala = true;
			}
		}
		if (!mala){
			res.push_back(fila[i]);
		}
	}
	return res;
}

// Función para buscar los ojos
Mat BuscarOjos(Mat imagen){
	Mat res = imagen;
	//imagen.copyTo(res);
	int mitad_x = imagen.rows / 2;
	int seguidos = 0;
	int tamaojo = 35, error = 5;
	vector <int> fila;
	int numero_ojos = 0;
	for (int i = 0; i < mitad_x; i++){
		for (int j = 0 + 20; j < imagen.cols - 20; j++){
			if (imagen.at<Vec3b>(i, j) == Vec3b(0, 0, 0)){
				seguidos++;
			}
			else {
				if (seguidos >= (tamaojo - error) && seguidos <= (tamaojo + error)){
					numero_ojos++;
				}

				seguidos = 0;
			}
		}
		if (numero_ojos == 2)
			fila.push_back(i);
		numero_ojos = 0;
	}

	vector<int> ojos;
	ojos = BuscarOjosProfundo(fila, res);
	if (ojos.size() != 0){
		rectangle(res, Point(0, ojos[0]), Point(imagen.cols - 1, ojos[ojos.size() - 1]), CV_RGB(0, 0, 255));
	}
	else
		cout << " No se reconocen ojos";
	return res;
}


bool ParecePupilaRGB(Mat im, int coor_x, int coor_y){
	bool parece_negro = false;
	double coo1 = im.at<Vec3b>(coor_x, coor_y)[0]; //R
	double coo2 = im.at<Vec3b>(coor_x, coor_y)[1]; //G
	double coo3 = im.at<Vec3b>(coor_x, coor_y)[2]; //B

	if (coo1 < 70 && coo2 < 60 && coo3 < 70)
		parece_negro = true;

	return parece_negro;
}

bool PareceBlancoOjoRGB(Mat im, int coor_x, int coor_y){
	bool parece_blanco = false;
	double coo1 = im.at<Vec3b>(coor_x, coor_y)[0]; //R
	double coo2 = im.at<Vec3b>(coor_x, coor_y)[1]; //G
	double coo3 = im.at<Vec3b>(coor_x, coor_y)[2]; //B

	if (coo1>200 && coo2>190 && coo3>190)
		parece_blanco = true;

	return parece_blanco;
}

//Le paso la imagen de piel 
Mat BuscaOjosCris(Mat piel_b_n, Mat original_recor, bool &encontrado){
	Mat aux = piel_b_n;
	for (int i = 0; i < piel_b_n.rows; i++){
		for (int j = 0; j < piel_b_n.cols; j++){
			if (piel_b_n.at<Vec3b>(i, j) == Vec3b(0, 0, 0))
				if (ParecePupilaRGB(original_recor, i, j))
					aux.at<Vec3b>(i, j) = Vec3b(100, 10, 100); 
		}
	}
	//Ya tengo pintado en morado la zona posible de pupila.
	//Ahora tengo que encontradar dos zonas moradas a cierta distancia relativa
	bool primero = false;
	int fil = 0; 
	int col = 0;
	for (int  i= aux.rows/2; i > 0; i--){
		for (int j = 0; j < aux.cols; j++){
			if (aux.at<Vec3b>(i, j) == Vec3b(100, 10, 100) && !primero){ //punto bajo de la pupila más baja
				primero = true;
				//tengo que ver que efectivamente es una pupila. que tenga a derecha e izquierda algo blanco
				fil = i;
				col = j;
			}
		}
	}

	//Zonas blancas para verificar que es ojo
	for (int i = 0; i < piel_b_n.rows; i++){
		for (int j = 0; j < piel_b_n.cols; j++){
			if (piel_b_n.at<Vec3b>(i, j) == Vec3b(0, 0, 0))
				if (PareceBlancoOjoRGB(original_recor, i, j))
					aux.at<Vec3b>(i, j) = Vec3b(10, 100, 10); //pinto en verde las "zonas blancas"
		}
	}

	//Busco que en la línea que se ha encontrado como ojos, sea de verdad un ojo.
	//Que en su entorno tenga algún pixel "ParaceBlancoOjo"	que ahora será color 10,100,10
	bool es_bueno = false;
	for (int k = fil - 10; (k < fil + 10) && (0<k) && (k<aux.rows); k++){
		for (int l = col - 10; (l < col + 10) && (0<l) && (l<aux.cols); l++){
			if (aux.at<Vec3b>(k, l) == Vec3b(10, 100, 10) && !es_bueno){ //hay zona blanca ojo
				es_bueno = true;
				encontrado = true;
				line(aux, Point(0, fil), Point(aux.cols, fil), CV_RGB(0, 0, 255));
			}
		}
	}

	//Si no se han encontrado ojos en la mitad superior, buscando en la inferior haciendo busqueda de arriba a abajo
	bool primeroabajo = false; 
	if (!es_bueno){
		cout << "bajo" << endl;
		for (int i = aux.rows / 2; i < aux.rows; i++){
			for (int j = 0; j < aux.cols; j++){
				if (aux.at<Vec3b>(i, j) == Vec3b(100, 10, 100) && !primeroabajo){ //punto bajo de la pupila más baja
					primeroabajo = true;
					//tengo que ver que efectivamente es una pupila. que tenga a derecha e izquierda algo blanco
					fil = i;
					col = j;
					cout << "abajo" << endl;
				}
			}
		}
		bool es_buenoabajo = false;
		for (int k = fil - 10; (k < fil + 10) && (0 < k) && (k < aux.rows); k++){
			for (int l = col - 10; (l < col + 10) && (0 < l) && (l < aux.cols); l++){
				if (aux.at<Vec3b>(k, l) == Vec3b(10, 100, 10) && !es_buenoabajo){ //hay zona blanca ojo
					es_buenoabajo = true;
					encontrado = true;
					line(aux, Point(0, fil), Point(aux.cols, fil), CV_RGB(0, 0, 255));
				}
			}
		}
	}



	return aux;
}

Mat HastaEncontrarOjos(Mat piel_b_n, Mat original_recor){
	bool encontrado = false;
	Mat ojos = BuscaOjosCris(piel_b_n, original_recor, encontrado);
	if (!encontrado){
		cout << "estoy en el nuevo metodo. No he encontrado " << endl;
		//Recorto imagen por la izquierda y la derecha unos 10 píxeles.

	}
	return ojos;
}

int main(){
	//Leemos las imágenes que vamos a usar para las pruebas
	int numero_imagenes;
	string nombre_imagenes;
	vector<Mat> imagenes_caras;
	int flag_color = CV_32FC3;

	//Leemos las imágenes sacadas de una base de datos
	numero_imagenes = 70;
	nombre_imagenes = "imagenes/image_000";
	imagenes_caras = LeerImagenes(numero_imagenes, nombre_imagenes, flag_color);

	//2. Pasar de Color Carne a Negro
	int tolerancia;
	tolerancia = 70;
	vector<Mat> imagenes_sin_color_carne;
	//imagenes_sin_color_carne = DetectarRosa(imagenes_caras,tolerancia);


	//3. Sacar piel de las imágenes

	//3.1 Sacar piel
	vector<Mat> imagenes_salida;
	vector<Mat> imagenes_bnYCrCb;
	int contador = 0;
	//Para las imagenes que no reconocen cara
	vector<Mat> imagenes_caras_malas;
	vector<Mat> imagenes_salida_malas;
	vector<Mat> imagenes_bnYCrCb_malas;

	vector<Mat> imagenes_recortadas;
	vector<Mat> imagenes_gaussianas;

	int primero_filas = 0;
	int ultimo_filas = 0;
	int primero_col = 0;
	int ultimo_col = 0;
	//Busco la zona cuadrada de piel en cada imagen 
	for (int i = 0; i < numero_imagenes; i++){
		//Pasamos las imágenes de RGB a YCrCb y después a Blanco-Negro
		Mat aux = TransformarDeRGBaYCrCBYPasoABlancoNegro(imagenes_caras[i], 110, 80, 130, 145, 180);
		if (BuscarSiHayCara(aux)){

			//3.2 Recortar piel
			//Si hay cara pasamos a recortarla
			imagenes_bnYCrCb.push_back(aux);
			Mat salida = recortoCaraFILAS(aux, primero_filas, ultimo_filas);
			salida = recortoCaraCOL(salida, primero_col, ultimo_col);
			imagenes_salida.push_back(salida);

			//pintaI(imagenes_bnYCrCb[contador], "b/n piel");
			//pintaI(imagenes_salida[contador], "recortada");

			Mat recortada = RecortarImagen(imagenes_caras[i], primero_filas, ultimo_filas, primero_col, ultimo_col);
			imagenes_recortadas.push_back(recortada);
			//pintaI(imagenes_recortadas[contador], "recortada orgi");

			Mat ojos = HastaEncontrarOjos(salida, recortada);
			pintaI(ojos, "ojitos");

			//pintaI(imagenes_recortadas[contador], "recortada orgi");

			/*
			Mat imgaus5 = frecuenciaAlta(imagenes_recortadas[contador], 3.0);
			imagenes_gaussianas.push_back(imgaus5);
			pintaI(imagenes_gaussianas[contador], " gaussiano");
			*/
			contador++;
		}
		else{
			Mat malas = imagenes_caras[i];
			imagenes_caras_malas.push_back(malas);
			cout << "\nImagen " << i << " No se reconoce cara";
			//pintaI(aux, "No se reconoce la cara.");
		}
	}


	system("pause");
}
