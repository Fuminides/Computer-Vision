#Calculas las medias y varianzas de todas las clases
M = load('objetos');
numero_elementos_clase = 5; #Usar matriz si cada elemento tiene distinto numero.
format long g

#Me tomo una pequena licencia: se en que orden voy a poner las muestras,
#asi que me ahorro leer las labels y tener que hacer cosas mas complicadas
#para sacar los arrays de cada clase.
circulos = M(1:5,1:end-1);
rectangulos = M(6:10,1:end-1);
ruedas =M(11:15,1:end-1);
triangulos =M(16:20,1:end-1);
vagon = M(21:25,1:end-1);

circulos_medias = mean(circulos);
circulos_varianzas = var(circulos);
fid = fopen('Circulos medias.txt', 'wt');
fprintf(fid, '%f %f %f %f %f %f %f %f', circulos_medias, circulos_varianzas);
fclose(fid);
rectangulos_medias = mean(rectangulos);
rectangulos_varianzas = var(rectangulos);
fid = fopen('Rectangulos medias.txt', 'wt');
fprintf(fid, '%f %f %f %f %f %f %f %f', rectangulos_medias, rectangulos_varianzas);
fclose(fid);
ruedas_medias = mean(ruedas);
ruedas_varianzas = var(ruedas);
fid = fopen('Ruedas medias.txt', 'wt');
fprintf(fid, '%f %f %f %f %f %f %f %f', ruedas_medias, ruedas_varianzas);
fclose(fid);
triangulos_medias = mean(triangulos);
triangulos_varianzas = var(triangulos);
fid = fopen('Triangulos medias.txt', 'wt');
fprintf(fid, '%f %f %f %f %f %f %f %f', triangulos_medias, triangulos_varianzas);
fclose(fid);
vagon_medias = mean(vagon);
vagon_varianzas = var(vagon);
fid = fopen('Vagones medias.txt', 'wt');
fprintf(fid, '%f %f %f %f %f %f %f %f', vagon_medias, vagon_varianzas);
fclose(fid);