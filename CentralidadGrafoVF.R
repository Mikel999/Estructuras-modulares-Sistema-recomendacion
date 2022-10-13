library(igraph)
library(influenceR)
library(caret)
library(caTools)
library(class)
library(tidymodels)

#Leer fichero txt
data<-read.table("Datasets/user_rating.txt")
#Eliminar columna de fechas
data<-data[,-4]
#Asignar nombres a las columnas
colnames(data)<-c("ID_A", "ID_B", "Rating")
#Eliminar las filas con enlaces tipo A->A
data<-data[-which(data[,1]==data[,2]),]

#Seleccionar los primeros 250 indices de cada caso (rating 1 y -1) 
#despues de un numero aleatorio,  para mantener los datos balanceados
set.seed(9999)
r<-sample(100000,1)#Random number=13607L
index<-which(data$Rating==-1)[r:(r+249)]
index2<-which(data$Rating==1)[r:(r+249)]
indexes<-union(index, index2)
#Escoger los primeros 500 enlaces
data<- data[indexes,]

#Poner los pares en formato de vector para construir grafo
mat<-data[,1:2]
linkVec<-as.vector(t(mat))
#Lista de nodos unicos, para construir el grafo
uniqueNodes<-length(unique(linkVec))
#Secuencia de numeros para guardar el orden de pares de los ids de nodos
#para crear el grafo se necesitan identidades de nodos, por ello se 
#ha escogido utilizar la funcion rank, tiene que haber un orden en los IDs
rankVector<-rank(linkVec)

#Crear nuevo dataframe para ir añadiendo las caracteristicas por enlace
dataFrame<-data.frame(c1=numeric(), 
                      c2=numeric(), 
                      c3=numeric(), 
                      c4=numeric(), 
                      c5=numeric(), 
                      c6=numeric(),
                      c7=numeric())

#Ir quitando los enlaces de uno en uno para construir 
#las caracteristicas de centralidad del grafo sin cada par
#(los enlaces en formato de vector son dos numeros seguidos 1-2, 3-4...)
for (i in 0:(length(indexes)-1)){
  print(i)
  #Por cada par de enlaces, construir grafo con el resto
  edges<-rankVector[-(i*2+1):-(i*2+2)]
  g<-make_graph(edges, n=uniqueNodes, directed = TRUE)
  E(g)$weight<-data$Rating[-(i+1)]
  #Calcular las distintas centralidades del grafo
  c1<-centralization.degree(g, mode="all")$centralization
  c2<-centralization.degree(g, mode="in")$centralization
  c3<-centralization.degree(g, mode="out")$centralization
  c4<-alpha.centrality(g, alpha=0.9)
  c4<-centralize(c4, theoretical.max = 50, normalized = T)
  c5<-power_centrality(g,exponent =0.9)
  c5<-centralize(c5, theoretical.max = 10, normalized = T)
  c6<-betweenness(g, snap=T)
  c6<-centralize(c6,theoretical.max = 100000 ,normalized=T)
  c7<-subgraph_centrality(g)
  c7<-centralize(c7,theoretical.max = 1000 ,normalized=T)
  #Crear fila con las caracteristicas del centralizacion del enlace
  dataRow<-data.frame(c1=c1, 
                      c2=c2, 
                      c3=c3, 
                      c4=c4, 
                      c5=c5, 
                      c6=c6,
                      c7=c7)
  #Encadenar al dataframe la fila
  dataFrame<-rbind(dataFrame, dataRow)
}
#Juntar columna de ratings de cada enlace como clase
dataFrame$Class<-data[,3]

########################################################################
###################### CLASIFICACION ###################################
########################################################################

#Separar base de datos en test train
set.seed(9999) 
sample<-sample.split(dataFrame$Class, SplitRatio = .80)
train<-subset(dataFrame, sample == TRUE)
test<-subset(dataFrame, sample == FALSE)

#Dataframe para guardar las metricas de cada modelo
metricas<-data.frame("Accuracy"=numeric(),
                     "Precision"=numeric(),
                     "Recall"=numeric(),
                     "F_Score"=numeric(),
                     "Mcc_Score"=numeric())

#Nombres para las filas del dataframe de metricas
names<-c("KNN", "NB", "RF", "DT", "NN")
#Nombres para llamar al metodo de entrenamiento desde un bucle
methodNames<-c("knn","naive_bayes","rf","rpart","nnet")
#Vector para guardar los valores del test de normalidad de cada modelo
shapiroVec<-rep(0,5)
#Dataframe para guardar los valores de accuracy de cada fold por modelo
#para test anova o kruskal
FoldAccuracyDF<-data.frame("A1"=numeric(),
                           "A2"=numeric(),
                           "A3"=numeric(),
                           "A4"=numeric(),
                           "A5"=numeric(),
                           "A6"=numeric(),
                           "A7"=numeric(),
                           "A8"=numeric(),
                           "A9"=numeric(),
                           "A0"=numeric())

#Metodo de control de validacion cruzada con k=10
trainC<-trainControl(method="cv", number=10)

#Entrenar los modelos, guardando las metricas de cada uno
for (i in 1:5){
  #Escoger la semilla para que todos los modelos tengan las mismas
  #particiones
  set.seed(100)
  #Entrenar los diferentes modelos, 
  modelTrain<- train(as.factor(Class) ~ ., data=train,
                     method=methodNames[i], trControl=trainC)
  #Predecir con el modelo entrenado las clases de testeo
  prediction <- predict(modelTrain, newdata = test, type = "raw")
  #Resultados mediante matriz de confusion
  tabla<-confusionMatrix(data=prediction,reference=as.factor(test$Class))
  #Obtener medidas de rendimiento: accuracy, precision, recall, fscore, mcc
  accuracy<- tabla$overall['Accuracy']
  precision <- tabla$byClass['Pos Pred Value']    
  recall <- tabla$byClass['Sensitivity']
  fScore <- tabla$byClass['F1']
  mccScore<-mcc(data=test, as.factor(Class),prediction)$.estimate
  
  #Crear dataframe de una fila con las metricas obtenidas por cada modelo
  dataRow<-data.frame("Accuracy"=accuracy,
                      "Precision"=precision,
                      "Recall"=recall,
                      "F_Score"=fScore,
                      "Mcc_Score"=mccScore)
  
  #Encadenar la fila al dataframe de metricas
  metricas<-rbind(metricas, dataRow)
  
  #Test de normalidad con el vector de resultados del accuracy
  FoldAccuracy<-modelTrain[["resample"]][["Accuracy"]]
  #Guardar el p valor del test de shapiro en el vector de normalidad
  shapiroVec[i]<-shapiro.test(FoldAccuracy)$p.value
  #Guardar los valores de accuracy en dataframe para el test
  #de anova o kruskal-wallis despues
  FoldAccuracyDF<-rbind(FoldAccuracyDF, FoldAccuracy)
  
  #Guardar las particiones del cross validation del primer modelo
  #para poder copiar las particiones en los modelos restantes
  #aparte de la semilla es necesario los parametros
  #index e indexout del traincontrol para mantener las particiones
  if (i==1){
    trainC<-trainControl(method="cv", number=10, 
                         index = modelTrain[["control"]][["index"]],
                         indexOut = modelTrain[["control"]][["indexOut"]])
  }
}
#Poner nombres de modelos a cada fila en el dataframe de las metricas
row.names(metricas)<-names
#Poner nombres a los distintos folds
colnames(FoldAccuracyDF)<-c("F1","F2","F3","F4","F5","F6","F7",
                            "F8","F9","F10")

#Comprobar si algun modelo tiene distribucion no normal
if(length(which(shapiroVec<=0.05))>1){
  print("No se puede asumir normalidad en los modelos.")
  #Como no se puede asumir normalidad en todos los modelos,
  #test de kruskal wallis
  krusk<-kruskal.test(list(unname(as.numeric(FoldAccuracyDF[1,])),
                           unname(as.numeric(FoldAccuracyDF[2,])),
                           unname(as.numeric(FoldAccuracyDF[3,])),
                           unname(as.numeric(FoldAccuracyDF[4,])),
                           unname(as.numeric(FoldAccuracyDF[5,]))))
  print(krusk)
}

#Siendo el p-valor tan pequeño, se rechaza la idea de que la 
#diferencia se debe al muestreo aleatorio y, en cambio, 
#se puede concluir que las poblaciones tienen distribuciones diferentes.

