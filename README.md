# Vision_Por_Computador

Participantes: 
Francisco Pérez Hernández: https://github.com/PacoPollos
Cristina Zuheros Montes: https://github.com/cristinazuhe

En este proyecto, hemos creado, en OpenCV con C++, un reconocedor de caras. Para este reconocedor hemos realizado distintas técnicas, las cuales se pueden ver en la presentación adjunta en el proyecto. 
La técnica que mejor nos ha funcionado ha sido transformar la imagen de RGB a YCrCb y de esta sacar las partes con piel, para poder tener un recorte de la cara. Seguidamente hemos buscado los ojos en la zona recortada, y una vez reconocidos estos hemos seleccionado que tenemos cara. 
Comparando la base de datos de imágenes usadas, en nuestro reconocedor y en el que implementa OpenCV con la técnica Haar, hemos obtenido para ambos reconocedores un 82% de acierto. Decir que hay algunas imágenes para ambos reconocedores que dan positivo siendo algún error, pero estos casos son muy pequeños.


