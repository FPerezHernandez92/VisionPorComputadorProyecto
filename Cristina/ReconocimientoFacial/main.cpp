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

bool pintar_imagenes = false;
bool escribir_imagen = false;
int cont = 0;

/*************************************************/
/*************** Funciones auxiliares ************/
/*************************************************/
void pintaI(Mat im, char ventana[]);

/* Función para leer imágenes */
vector<Mat> LeerImagenes(int numero_imagenes, string nombre_imagenes, int flag_color){
	vector<Mat> imagenes;
	for (int i = 1; i <= numero_imagenes; i++){
		string aux_nombre = nombre_imagenes + to_string(i) + ".jpg";
		Mat aux_imagen = imread(aux_nombre, flag_color);

		if (pintar_imagenes)
			pintaI(aux_imagen, "Imagen original");

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

void otropintaI(Mat im, string ventana, int c){

	imwrite("salida/" + ventana + to_string(c) + ".jpg", im);
}
/* Función para pintar imágenes */
void PintaImagenes(vector<Mat> imagenes_caras, string nombre_imagenes = "Salida", bool escribir_imagen_salida = false){
	for (int i = 0; i < imagenes_caras.size(); i++){
		string aux_nombre = nombre_imagenes + to_string(i);
		if (escribir_imagen_salida)
			imwrite("salida/" + to_string(i) + ".jpg", imagenes_caras[i]);
		else {
			namedWindow(aux_nombre, imagenes_caras[i].channels());
			imshow(aux_nombre, imagenes_caras[i]);
			cvWaitKey();
			destroyWindow(aux_nombre);
		}
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
Mat PasarANegro(Mat imagen, int tolerancia = 70){
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
vector<Mat> DetectarRosa(vector<Mat> imagenes, int tolerancia, int num){
	vector<Mat> imagenes_salida;
	for (int i = 0; i < num; i++){
		//int aux = rand() % imagenes.size();
		Mat salida;
		salida = PasarANegro(imagenes[i], tolerancia);
		imagenes_salida.push_back(salida);
	}

	//Pintamos las imágenes que vamos a usar para las pruebas
	if (pintar_imagenes)
		PintaImagenes(imagenes_salida, "Color carne a blanco-negro");

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
	//	pintaI(salida, "YCrCb");

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
			if ((imagen.at<Vec3b>(fila[i], ojo1 + j) != Vec3b(0, 0, 0)) && (imagen.at<Vec3b>(fila[i], ojo2 + j) != Vec3b(0, 0, 0))){
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
void BuscarOjos(Mat imagen, int &contador_ojos_no_reconocidos, vector<Mat> &imagenes_ojos_identificados, vector<Mat> &imagenes_ojos_no_identificados){
	Mat res;
	imagen.copyTo(res);
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
		if (pintar_imagenes) pintaI(res, "Primer Buscador de ojos");
		imagenes_ojos_identificados.push_back(res);
	}
	else{
		contador_ojos_no_reconocidos++;
		imagenes_ojos_no_identificados.push_back(res);
	}
}

vector<int> LimpiarVector(vector<int> &filas){
	vector<int> res;
	bool elimino = true;
	for (int i = 0; i < filas.size(); i++){
		res.push_back(filas[i]);
	}
	while (elimino){
		elimino = false;
		for (int i = 1; i < res.size() - 1; i++){
			if (abs(res[i] - res[i - 1]) < 4){

			}
			else if (abs(res[i] - res[i + 1]) < 4){

			}
			else{
				elimino = true;
				res[i] = 999999999;
			}
		}
		if (elimino){
			vector<int> auxres;
			for (int i = 0; i < res.size(); i++){
				if (res[i] < 99999){
					auxres.push_back(res[i]);
				}
			}
			res.clear();
			for (int i = 0; i < auxres.size(); i++){
				res.push_back(auxres[i]);
			}
		}
	}


	return res;
}

void BuscarOjos3(Mat imagen){
	vector<int> filascandidatas;
	int distancia = imagen.cols / 2;
	int seguidos = 0;
	for (int i = 0; i < imagen.rows; i++){
		seguidos = 0;
		for (int j = 15, h = distancia + 15; h < imagen.cols; j++, h++){
			if ((imagen.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) && (imagen.at<Vec3b>(i, h) == Vec3b(0, 0, 0))){
				seguidos++;
			}
			else{
				if (seguidos > 20 && seguidos < 40){
					filascandidatas.push_back(i);
				}
				seguidos = 0;
			}
		}
	}
	cout << endl;
	for (int i = 0; i < filascandidatas.size(); i++){
		cout << filascandidatas[i] << " ";
	}
	cout << endl;

	if (filascandidatas.size() > 1){
		filascandidatas = LimpiarVector(filascandidatas);
		cout << "Aplico preprocesado: " << endl;
		for (int i = 0; i < filascandidatas.size(); i++){
			cout << filascandidatas[i] << " ";
		}
		cout << endl;
	}
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
Mat BuscaOjos2(Mat imagen_cara, Mat piel_b_n, Mat original_recor, bool &encontrado, int filas_recortadas_arriba, int col_recortadas_izquierda){
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
	for (int i = aux.rows / 2; i > 0; i--){
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
	for (int k = fil - 20; (k < fil + 20) && (0<k) && (k<aux.rows); k++){
		for (int l = col - 20; (l < col + 20) && (0<l) && (l<aux.cols); l++){
			if (aux.at<Vec3b>(k, l) == Vec3b(10, 100, 10) && !es_bueno){ //hay zona blanca ojo
				es_bueno = true;
				encontrado = true;
				line(imagen_cara, Point(col_recortadas_izquierda, fil + filas_recortadas_arriba), Point(col_recortadas_izquierda + aux.cols, fil + filas_recortadas_arriba), CV_RGB(0, 0, 255), 2);
				rectangle(imagen_cara, Point(col_recortadas_izquierda, filas_recortadas_arriba), Point(col_recortadas_izquierda + aux.cols, filas_recortadas_arriba + aux.rows), CV_RGB(0, 255, 0), 3);
				line(aux, Point(0, fil), Point(aux.cols, fil), CV_RGB(0, 0, 255));
			}
		}
	}

	//Si no se han encontrado ojos en la mitad superior, buscando en la inferior haciendo busqueda de arriba a abajo
	bool primeroabajo = false;
	if (!es_bueno){
		for (int i = aux.rows / 2; i < aux.rows; i++){
			for (int j = 0; j < aux.cols; j++){
				if (aux.at<Vec3b>(i, j) == Vec3b(100, 10, 100) && !primeroabajo){ //punto bajo de la pupila más baja
					primeroabajo = true;
					//tengo que ver que efectivamente es una pupila. que tenga a derecha e izquierda algo blanco
					fil = i;
					col = j;
				}
			}
		}
		bool es_buenoabajo = false;
		for (int k = fil - 20; (k < fil + 20) && (0 < k) && (k < aux.rows); k++){
			for (int l = col - 20; (l < col + 20) && (0 < l) && (l < aux.cols); l++){
				if (aux.at<Vec3b>(k, l) == Vec3b(10, 100, 10) && !es_buenoabajo){ //hay zona blanca ojo
					es_buenoabajo = true;
					encontrado = true;
					line(imagen_cara, Point(col_recortadas_izquierda, fil + filas_recortadas_arriba), Point(col_recortadas_izquierda + aux.cols, fil + filas_recortadas_arriba), CV_RGB(0, 0, 255), 2);
					rectangle(imagen_cara, Point(col_recortadas_izquierda, filas_recortadas_arriba), Point(col_recortadas_izquierda + aux.cols, filas_recortadas_arriba + aux.rows), CV_RGB(0, 255, 0), 3);
					line(aux, Point(0, fil), Point(aux.cols, fil), CV_RGB(0, 0, 255));
				}
			}
		}
	}

   //  pintaI(aux, "Segundo buscador de ojos");
	if (escribir_imagen) otropintaI(aux, "SegundoBuscadorDeOjos", cont);
	cont++;

	return imagen_cara;
}

Mat HastaEncontrarOjos(Mat imagen_cara, Mat piel_b_n, Mat original_recor, int &ojos_encontrados,
	int filas_recortadas_arriba, int col_recortadas_izquierda){
	bool encontrado = false;
	Mat ojos = BuscaOjos2(imagen_cara, piel_b_n, original_recor, encontrado, filas_recortadas_arriba, col_recortadas_izquierda);
	if (!encontrado){
		Mat recortada_b_n;
		Mat recortada_orig;
		piel_b_n.copyTo(recortada_b_n);
		original_recor.copyTo(recortada_orig);
		//Recorto imagen por la izquierda y la derecha unos 10 píxeles.
		int colum = piel_b_n.cols;
		int primera_colum = 10;
		int ultima_colum = colum - 10;
		int contador_while = 0;
		while (contador_while < 8 && !encontrado){
			if ((primera_colum < colum) && (ultima_colum > 0) && (primera_colum < ultima_colum)){
				recortada_b_n = RecortarImagen(piel_b_n, 0, piel_b_n.rows, primera_colum, ultima_colum);
				recortada_orig = RecortarImagen(original_recor, 0, piel_b_n.rows, primera_colum, ultima_colum);
				ojos = BuscaOjos2(imagen_cara, recortada_b_n, recortada_orig, encontrado, filas_recortadas_arriba, col_recortadas_izquierda);
				contador_while++;
				primera_colum = primera_colum + 10;
				ultima_colum = ultima_colum - 10;
			}
			else
				contador_while = 10;
		}

	}
	if (encontrado)
		ojos_encontrados++;
	return ojos;
}

//2. Pasar de Color Carne a Blanco-Negro
void PasarDeColorCarneABlancoNegro(int tolerancia, int numero_imagenes, vector<Mat> imagenes_caras){
	vector<Mat> imagenes_sin_color_carne;
	int num_imagenes = numero_imagenes; //Número de imágenes para mostrar
	cout << "-------------------------> 2 Pasando de color carne a blanco-negro: " << endl;
	imagenes_sin_color_carne = DetectarRosa(imagenes_caras, tolerancia, num_imagenes);
	//Para evitar problemas de memoria voy vaciando lo que no nos servirá en un futuro
	imagenes_sin_color_carne.clear();
}


//3.1 y 3.2 Vamos a sacar la piel de las imagenes y a recortar la piel
void SacarPielYRecortarPiel(vector<Mat> imagenes_caras, vector<Mat> &imagenes_caras_buenas,
	vector<Mat> &imagenes_recortadas, vector<Mat> &imagenes_color_recortadas,
	vector<int> &filas_recortadas_arriba/*primero_filas*/,
	vector<int> &col_recortadas_izquierda /*primero_col*/, vector<Mat> &imagenes_malas){
	//3.1 Sacar piel
	vector<Mat> imagenes_caras_malas;
	int primero_filas = 0, ultimo_filas = 0, primero_col = 0, ultimo_col = 0, contadormalas = 0, contador = 0;
	Mat salida;

	//Busco la zona cuadrada de piel en cada imagen 
	for (int i = 0; i < imagenes_caras.size(); i++){
		//Pasamos las imágenes de RGB a YCrCb y después a Blanco-Negro
		if (i == 0) cout << "-------------------------> 3.1 y 3.2 Pasando de RGB a YCrCb y a Blanco-Negro: " << endl << "Total de imagenes: " << imagenes_caras.size() << endl;
		cout << " " << i;
		Mat aux = TransformarDeRGBaYCrCBYPasoABlancoNegro(imagenes_caras[i], 0, 110, 140, 140, 170);
		//pintaI(aux, "YCrCb a blanco-negro");
		if (BuscarSiHayCara(aux)){
			//3.2 Recortar piel
			//Si hay cara pasamos a recortarla
			salida = recortoCaraFILAS(aux, primero_filas, ultimo_filas);
			salida = recortoCaraCOL(salida, primero_col, ultimo_col);
			imagenes_recortadas.push_back(salida);

			//pintaI(imagenes_recortadas[contador], "Recortada");

			col_recortadas_izquierda.push_back(primero_col);
			filas_recortadas_arriba.push_back(primero_filas);
			Mat color_recortada = RecortarImagen(imagenes_caras[i], primero_filas, ultimo_filas, primero_col, ultimo_col);
			imagenes_color_recortadas.push_back(color_recortada);
		    //pintaI(color_recortada, "ColorRecortada");
			imagenes_caras_buenas.push_back(imagenes_caras[i]);
			contador++;
		}
		else{
			Mat malas = imagenes_caras[i];
			cout << " (" << i << " No cara)";
		   // pintaI(aux, "No se reconoce la cara.");
			imagenes_caras_malas.push_back(aux);
			contadormalas++;
			imagenes_malas.push_back(imagenes_caras[i]);
		}
	}

	double porcentaje = (contadormalas / (imagenes_caras.size()*1.0)) * 100;
	cout << "\n\nNo se ha reconocido cara en " << contadormalas << " imagenes de " << imagenes_caras.size() << " ,un porcentaje de: " << porcentaje << "% de fallos" << endl;
	imagenes_caras_malas.clear();
}

//3.3 Vamos a pasar el primer buscador de ojos sobre las imagenes recortadas
void PrimerBuscadorDeOjos(vector<Mat> imagenes_recortadas){
	cout << "\n-------------------------> 3.3 Buscando ojos en imagenes recortadas: " << endl;
	vector<Mat> imagenes_ojos_identificados, imagenes_ojos_no_identificados;
	int contador_ojos_no_reconocidos = 0;
	Mat salidaaux;
	cout << "Total de imagenes: " << imagenes_recortadas.size() << endl;
	for (int i = 0; i < imagenes_recortadas.size(); i++){
		//3.3 Buscar ojos en una cara
		cout << " " << i;
		BuscarOjos(imagenes_recortadas[i], contador_ojos_no_reconocidos, imagenes_ojos_identificados, imagenes_ojos_no_identificados);
	}
	double porcentaje = (contador_ojos_no_reconocidos / (imagenes_recortadas.size()*1.0)) * 100;
	cout << "\n\nNo se han reconocido ojos en " << contador_ojos_no_reconocidos << " imagenes de " <<
		imagenes_recortadas.size() << " ,un porcentaje de: " << porcentaje << "% de fallos" << endl;

}

//4. Aplicar filtro gaussiano
void AplicarFiltroGaussiano(vector<Mat> imagenes_color_recortadas){
	cout << "\n-------------------------> 4 Aplicando filtro gaussiano: (Proceso lento)" << endl;
	vector<Mat> imagenes_gaussianas;
	cout << "Total de imagenes: " << imagenes_color_recortadas.size() << endl;
	for (int i = 0; i < imagenes_color_recortadas.size(); i++){
		cout << " " << i;
		Mat imgaus5 = frecuenciaBaja(imagenes_color_recortadas[i], 3.0);
		imagenes_gaussianas.push_back(imgaus5);
	    pintaI(imagenes_gaussianas[i], " Filtro Gaussiano");
	}
	cout << endl;
}

//5. Segundo buscador de ojos
void SegundoBuscadorDeOjos(vector<Mat> imagenes_caras_buenas, vector<Mat> imagenes_recortadas, vector<Mat> imagenes_color_recortadas,
	vector<int> filas_recortadas_arriba, vector<int> col_recortadas_izquierda){
	cout << "\n-------------------------> 5 Segundo Reconocedor de ojos: " << endl;
	int ojos_encontrados = 0;
	cout << "Total de imagenes: " << imagenes_recortadas.size() << endl;
	for (int i = 0; i < imagenes_recortadas.size(); i++){
		cout << " " << i;
		Mat ojos = HastaEncontrarOjos(imagenes_caras_buenas[i], imagenes_recortadas[i], imagenes_color_recortadas[i], ojos_encontrados,
			filas_recortadas_arriba[i], col_recortadas_izquierda[i]);
	    pintaI(ojos, "Final");
		if(escribir_imagen) otropintaI(ojos, "Final", i);
	}
	double porcentaje = (ojos_encontrados / (imagenes_recortadas.size()*1.0)) * 100;
	cout << "\n\nSe han encontrado: " << ojos_encontrados << " ojos de " << imagenes_recortadas.size() << " ,un porcentaje de: " << porcentaje << "% de acierto" << endl;

}

Mat PasandoB_N(Mat ima){
	Mat im;
	ima.copyTo(im);
	for (int i = 0; i < im.rows; i++){
		for (int j = 0; j < im.cols; j++){
			if (im.at<double>(i, j) > 45)
				im.at<double>(i, j) = 255;
			else
				im.at<double>(i, j) = 0;
		}
	}
	return im;
}

Mat CuartoBuscadorDeOjos(Mat im, int dimx, int dimy, int divi){
	Mat res;
	im.copyTo(res);
	int mejori = 0, mejorj = 0;
	double mejormedia = 0.0;
	//cout << endl << "6.2 Comienzo recortado. rows: " << im.rows << " cols: " << im.cols << endl;
	double auxmedia;
	int media = 0;
	int total = (dimy * (2 * divi));
	for (int i = 0; i < im.rows - dimy; i++){
		for (int j = 0; j < im.cols - dimx; j++){

			for (int h = 0; h < dimy; h++){
				for (int k = 0; k < dimx; k++){
					if ((k < divi / 3)){
						//cout << "i+h: " << i + h << " j+k: " << j + k << " val: " << im.at<float>(i + h, j + k) << endl;
						if (im.at<double>(i + h, j + k) > 250){
							media++;
						}
					}
				}
			}
			auxmedia = (media / (total*1.0)) * 100;
			if (mejormedia < auxmedia){
				mejormedia = auxmedia;
				mejori = i;
				mejorj = j;
			}
			media = 0;
		}
	}
	//cout << "mejormedia: " << mejormedia << " i: " << mejori << " j: " << mejorj << " i2: " << mejori+dimy << " j2: " << mejorj+dimx << endl;
	rectangle(res, Point(mejorj, mejori), Point(mejorj + dimx, mejori + dimy), CV_RGB(0, 0, 255), 3);
	return res;
	//pintaI(res, "res");
}


Mat QuintoBuscadorDeOjos(Mat mascara, Mat im){
	Mat res;
	im.copyTo(res);
	int mejori = 0, mejorj = 0;
	double mejormedia = 0.0;
	double auxmedia;
	int media = 0;
	int dimy = mascara.rows;
	int dimx = mascara.cols;
	int total = dimy*dimx;
	for (int i = 0; i < im.rows - dimy; i++){
		for (int j = 0; j < im.cols - dimx; j++){

			for (int h = 0; h < dimy; h++){
				for (int k = 0; k < dimx; k++){
					if (im.at<double>(i + h, j + k) == (mascara.at<double>(h,k))){
						media++;
					}
				}
			}
			auxmedia = (media / (total*1.0)) * 100;
			if (mejormedia < auxmedia){
				mejormedia = auxmedia;
				mejori = i;
				mejorj = j;
			}
			media = 0;
		}
	}
	//cout << "mejormedia: " << mejormedia << " i: " << mejori << " j: " << mejorj << " i2: " << mejori + dimy << " j2: " << mejorj + dimx << endl;
	rectangle(res, Point(mejorj, mejori), Point(mejorj + dimx, mejori + dimy), CV_RGB(0, 0, 255), 1);
	return res;
}

//6. Sacar ojos a partir de filtro Gaussiano
void SacarOjosAPartirDeFiltroGaussianoPrimero(vector<Mat> imagenes){
	cout << "-------------------------> 6 Aplicando filtro gaussiano: " << endl;
	vector<Mat> imagenes_gaussianas;
	cout << "Total de imagenes: " << imagenes.size() << endl;
	for (int i = 0; i < imagenes.size(); i++){
		cout << "Imagen: " << i << " espere por favor, el proceso es lento" << endl;
		Mat imgaus5 = frecuenciaAlta(imagenes[i], 5.0);
		Mat aux = PasandoB_N(imgaus5);
		int dimx = 160, dimy = 30, divi = 55;
		aux = CuartoBuscadorDeOjos(aux, dimx, dimy, divi);
		pintaI(aux, "4 Buscador");
	}
	cout << endl;
}

//7. Sacar ojos a partir de filtro Gaussiano
void SacarOjosAPartirDeFiltroGaussianoSegundo(vector<Mat> imagenes){
	cout << "\n-------------------------> 7 Aplicando filtro gaussiano: " << endl;
	vector<Mat> imagenes_gaussianas;

	cout << "Tratando la primera imagen" << endl;
	Mat aux;
	imagenes[0].copyTo(aux);
	aux = aux.colRange(520,708);
	aux = aux.rowRange(264, 296);
	aux = frecuenciaAlta(aux, 5.0);
	aux = PasandoB_N(aux);
	cout << "Total de imagenes: " << imagenes.size() << endl;
	for (int i = 1; i < imagenes.size(); i++){
		cout << "Imagen: " << i << " espere por favor, el proceso es lento" << endl ;
		Mat imgaus5 = frecuenciaAlta(imagenes[i], 5.0);
		Mat aux2 = PasandoB_N(imgaus5);
		Mat salida = QuintoBuscadorDeOjos(aux, aux2);
		pintaI(salida, "5 Buscador");
	}
	cout << endl;
}

int BuenasOpenCV(vector<Mat> imagenes_caras){
	// Load Face cascade (.xml file)
	CascadeClassifier face_cascade;
	face_cascade.load("haarcascade_frontalface_alt.xml");

	// Detect faces
	std::vector<Rect> faces;
	int contador_supuestas_buenas = 0;
	for (int i = 0; i < imagenes_caras.size(); i++){
		face_cascade.detectMultiScale(imagenes_caras[i], faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		Mat auxi;
		imagenes_caras[i].copyTo(auxi);
		// Draw circles on the detected faces
		for (int i = 0; i < faces.size(); i++){
			Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			ellipse(auxi, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		}
		pintaI(auxi, "cara");
		if (faces.size() == 1){
			contador_supuestas_buenas++;
		}

	}
	return contador_supuestas_buenas;
}


int main(){
	//Leemos las imágenes que vamos a usar para las pruebas
	int numero_imagenes, flag_color = CV_32FC3;
	string nombre_imagenes;
	vector<Mat> imagenes_caras;

	//Leemos las imágenes sacadas de una base de datos
	numero_imagenes = 195;
	nombre_imagenes = "imagenes_malas/image_000";
	cout << "-------------------------> 1 Leyendo imagenes: " << endl;
	imagenes_caras = LeerImagenes(numero_imagenes, nombre_imagenes, flag_color);

	//2. Pasar de Color Carne a Blanco-Negro
	//int tolerancia = 70;
	//PasarDeColorCarneABlancoNegro(tolerancia, numero_imagenes, imagenes_caras);

	//3. Sacar piel de las imágenes
	//3.1 y 3.2 Sacar piel y recortar piel
	vector<int> filas_recortadas_arriba, col_recortadas_izquierda;
	vector<Mat> imagenes_recortadas, imagenes_color_recortadas, imagenes_caras_buenas, imagenes_malas;

	SacarPielYRecortarPiel(imagenes_caras, imagenes_caras_buenas, imagenes_recortadas, imagenes_color_recortadas, filas_recortadas_arriba, col_recortadas_izquierda, imagenes_malas);
	
	/*
	//Meto en un vector las imagenes que no han sido reconocidas (en b/n-->1canal)
	vector<Mat> imagenes_malas_bn;
	for (int i = 0; i < imagenes_malas.size(); i++){
		Mat aux;
		imagenes_malas[i].copyTo(aux);
		cvtColor(imagenes_malas[i], aux, CV_BGR2GRAY);
		imagenes_malas_bn.push_back(aux);
	}*/

	//3.3 Vamos a pasar a aplicar el primer buscar de ojos que hemos realizado
	//PrimerBuscadorDeOjos(imagenes_recortadas);

	//BuscarOjos3(salida);

	//4. Filtro Gaussiano
	//AplicarFiltroGaussiano(imagenes_caras);


	//En imagenes_caras_buenas tengo las imagenes que se han reconocido como que tienen cara.
	//El orden será el mismo que hay en imagenes_recortadas
	//5. Segundo buscador de ojos
	SegundoBuscadorDeOjos(imagenes_caras_buenas, imagenes_recortadas, imagenes_color_recortadas, filas_recortadas_arriba, col_recortadas_izquierda);

	//6. Sacar ojos a partir del filtro gaussiano
	/*
	flag_color = 0;
	numero_imagenes = 10;
	nombre_imagenes = "imagenes/image_000";
	cout << "\n-------------------------> 6 Leyendo imagenes blanco y negro \n" << endl;
	imagenes_caras.clear();
	imagenes_caras = LeerImagenes(numero_imagenes, nombre_imagenes, flag_color);


	SacarOjosAPartirDeFiltroGaussianoPrimero(imagenes_caras);

	SacarOjosAPartirDeFiltroGaussianoSegundo(imagenes_caras);*/

	//7- Usando OpenCV
	/*
	int contador_supuestas_buenas = BuenasOpenCV(imagenes_caras);
	double porcentaje = (contador_supuestas_buenas * 100) / (imagenes_caras.size());
	cout << "Buenas de OpenCV: " << contador_supuestas_buenas << ", un porcentaje de: " << porcentaje << "%" <<  endl;
	
	*/


	cout << endl << endl;
	system("pause");
}
