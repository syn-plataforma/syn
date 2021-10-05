# General

El proyecto SYN aborda varias de las problemáticas y limitaciones científico-técnicas existentes en el ámbito tecnológico relacionado con el mantenimiento software, con la intención de proponer una solución novedosa que mejore el estado actual del arte, y que permita implementar dicha solución en los sistemas de seguimiento de incidencias.

Dichas problemáticas y limitaciones científico-técnicas sobre las que se pretende investigar y plantear una solución se pueden clasificar en los siguientes puntos:
	
- Detección de incidencias duplicadas.
- Recuperación de incidencias similares.
- Detección de incidencias clasificadas de forma errónea.
- Detección de incidencias priorizadas de forma errónea.
- Detección de incidencias asignadas de forma errónea.

El objetivo principal de SYN, es crear un sistema avanzado de gestión de incidencias software, para lo que han desarrollado algoritmos híbridos basados en Redes Neuronales con mecanismos de atención, así como de Transfer Learning para la creación de un novedoso modelo que aplique una perspectiva holística a la gestión de incidencias en sistemas complejos de Software Infrastructure and Application Management.

Debido a que el presente proyecto se alberga dentro del campo de procesamiento de lenguaje natural y, a que el lenguaje de manera intrínseca exhibe propiedades sintácticas que naturalmente combinan palabras con frases formando estructuras secuenciales complejas, para materializar el sistema desarrollado, se han aplicado algoritmos de Inteligencia Artificial capaces de preservar la información secuencial. En concreto se han estudiado y empleado, **redes Tree-LSTM** (Tai et al., 2015), un tipo de red neuronal recurrente de alta complejidad adaptada a topologías estructuradas en forma de árbol como el análisis lingüístico, con **mecanismos de atención**. Este tipo de redes han obtenido excelentes resultados en tareas como predecir la relación semántica de dos oraciones y la clasificación de sentimiento.

El núcleo del sistema está constituido por un modelo neurosimbólico que, combinando redes neuronales y reglas simbólicas, es capaz de capturar tanto el conocimiento del programador como las evaluaciones de incidencias pasadas, de forma que es posible incorporar lo anteriormente aprendido y obtener una mejora sustancial en la resolución de las incidencias. Es un sistema altamente escalable, personalizable y dinámico capaz de adaptarse a las distintas casuísticas de cualquier entorno de proyecto de investigación (lenguaje utilizado, equipo de desarrollo, ámbito específico de los proyectos, etc.). Estas capacidades, además de suponer una ventaja competitiva para la empresa, facilitarán su utilización en ulteriores trabajos de investigación y desarrollo llevados a cabo por parte de la comunidad científica.

Una vez finalizado el proyecto, y analizados los resultados obtenidos, se inició una **nueva iteración del proyecto** en la que se propuso una **nueva solución**, con la que intentar superar los problemas encontrados durante la ejecución del presente proyecto de investigación. La nueva solución, se basó en la **utilización de metodologías de modelos de recuperación basadas en codebooks**, y de los propios codebooks como base de la representación textual sobre la que construir el sistema.
La base en la que se fundamentaba la hipótesis de que esta nueva solución, permitiría superar algunos de los problemas encontrados, se basaba en los excelentes resultados obtenidos por las metodologías basadas en codebooks en sistemas de análisis de imagen y de vídeo (X. Zhang et al., 2018), (Hu & Lu, 2018), (Kordopatis-Zilos et al., 2017), donde se utilizaban para generalizar las características extraídas por diferentes métodos.
Para comprobar la validez de esta hipótesis se creó un algoritmo original que seguía la siguiente metodología:
1.	Se construye un embedding a partir de un cierto corpus.
2.	Se construye el libro de códigos utilizando el mismo corpus, basándose en algoritmos de clustering.
3.	Con el libro de códigos construido, se transforma cada texto que se quiera tratar en la tarea en una secuencia de codewords.
4.	Con dicha secuencia y empleando los métodos propios del campo de la recuperación documental y del aprendizaje automático, se construye para cada tarea un sistema de recuperación o de clasificación según la naturaleza de esta.


# Instrucciones para crear los conjuntos de datos iniciales procedentes de Mining Software Repositories

1. Descargar los conjuntos de datos existentes en la página [Duplicate Bug Datasets (MSR 2014)][msr_repository] y seguir las instrucciones existentes en dicha página.
2. Importar las colecciones en MongoDB.

La siguiente tabla muestra el total de colecciones y su número de registros para cada base de datos.

| Base de datos | Colección | Número de Registros |
| ------------- | --------- | ------------------- |
| eclipse       | initial   |             423.559 |
| eclipse       | clear     |             361.006 |
| eclipse       | pairs     |             271.098 |
| netBeans      | initial   |             237.142 |
| netBeans      | clear     |             216.715 |
| netBeans      | pairs     |             238.584 |
| openOffice    | initial   |             123.865 |
| openOffice    | clear     |              98.070 |
| openOffice    | pairs     |             152.872 |

# Instrucciones para crear los ficheros de propiedades

1. Clonar el proyecto.
2. Cambiar al directorio el proyecto.
3. Crear un entorno virtual.
4. Activar el entorno virtual.
5. Instalar las dependencias.
6. Crear una copia del fichero 'definitions.sample.py' y renombrarla a 'definitions.py'.
7. Editar el fichero y establecer el valor de la propiedad SYN_ENV a 'stage'.
8. Crear


~~~
    git clone https://github.com/syn-plataforma/syn.git
    cd syn    
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    cp definitions.sample.py definitions.py
~~~

# Instrucciones para crear los conjuntos de datos iniciales procedentes de Bugzilla

1. Ejecutar el módulo que utiliza la [API de Bugzilla][bugzilla_api] para descargar las incidencias para cada año.
2. Concatenar todas las colecciones MongoDB de cada año para crear una única colección.
3. Seleccionar los campos presentes en las colecciones de Mining Software Repositories.
4. Generar los pares de incidencias similares y no similares.

~~~
    python3 -m syn.data.collect.bugzilla.create_all_bugzilla_mongodb_collection &
    python3 -m syn.data.collect.concatenate_mongodb_collections &
    python3 -m syn.data.select.bugzilla.useful_data &
    python3 -m syn.data.construct.bugzilla.generate_pairs &
    python3 -m syn.data.construct.bugzilla.generate_pairs_from_indirect_relations &
~~~

# Instrucciones para crear los conjuntos de datos iniciales procedentes de Gerrit

1. Ejecutar el módulo que utiliza la [API de Gerrit][gerrit_api] para descargar las revisiones de código para cada año del proyecto Eclipse.
2. Concatenar todas las colecciones MongoDB de cada año para crear una única colección.
3. Seleccionar las revisiones de código cuyo campo “subject” empiece por “Bug bug_id”, como por ejemplo “Bug 486901 …”.
4. Fusionar estas revisiones de código con las incidencias del proyecto Eclipse de Bugzilla.
5. Encontrar las revisiones de código similares. 
6. Generar los pares de incidencias similares y no similares.

~~~
    python3 -m syn.data.collect.gerrit.create_all_gerrit_mongodb_collections &
    python3 -m syn.data.collect.gerrit.concatenate_all_gerrit_mongodb_collections &
    python3 -m syn.data.select.gerrit.useful_data &
    python3 -m syn.data.integrate.gerrit.merge_data &
    python3 -m syn.data.construct.gerrit.find_similar_issues &
    python3 -m syn.data.construct.gerrit.generate_pairs &
~~~

# Instrucciones para crear los conjuntos de datos para entrenamientos y pruebas

1. Crear las colecciones MongoDB con la información textual normalizada para todos los corpus (Eclipse, NetBeans, OpenOffice y Bugzilla).
2. Crear índice en MongoDB para el campo 'creation_ts'.
3. Descargar los word embeddings pre-entrenados para GloVe, Word2Vec y fastText.
4. Construir los vocabularios.
5. Entrenar los word embeddings.
6. Filtrar los word embeddings.
7. Codificar la información estructurada de los conjuntos de datos.
8. Codificar las etiquetas de las tareas para detectar incidencias duplicadas y similares.
9. Construir los conjuntos de datos de entrenamiento, validación y pruebas.
10. Construir los conjuntos de datos de entrenamiento y pruebas para la asignación de incidencias basada en ocupación y adecuación de los desarrolladores.



python -m syn.data.clean.build_normalized_clear --db_name openOffice --collection_name clear --output_collection_name normalized_clear --max_num_tokens 0 --architecture codebooks

~~~
    python3 -m syn.data.clean.build_normalized_clear --db_name eclipse --collection_name clear --output_collection_name normalized_clear --max_num_tokens 0 --architecture codebooks &
    python3 -m syn.data.clean.build_normalized_clear --db_name netBeans --collection_name clear --output_collection_name normalized_clear --max_num_tokens 150 --architecture codebooks &
    python3 -m syn.data.clean.build_normalized_clear --db_name openOffice --collection_name clear --output_collection_name normalized_clear --max_num_tokens 0 --architecture codebooks &
    python3 -m syn.data.clean.build_normalized_clear --db_name bugzilla --collection_name clear --output_collection_name normalized_clear --max_num_tokens 0 --architecture codebooks &
    python3 -m syn.model.build.common.create_all_mongodb_index &
    python3 -m scripts.download --no_download_corenlp --no_download_srparser --no_download_mongodb &
    python3 -m syn.model.build.common.build_all_vocabs &
    python3 -m syn.model.build.common.train_all_word_embeddings &
    python3 -m syn.model.build.common.filter_all_word_embeddings &
    python3 -m syn.model.build.common.encode_all_strurctured_data &
    python3 -m syn.model.build.common.encode_all_pairs_labels &
    python3 -m syn.model.build.common.build_all_datasets &
    python3 -m syn.model.build.codebooks.build_custom_assignation_dataset &
~~~

# Instrucciones para entrenar los modelos

1. Entrenar el modelo.
2. Evaluar el modelo.
3. Realizar hiper-búsqueda

~~~
    python3 -m syn.model.build.codebooks.train --task prioritization --corpus openOffice --embeddings_model glove --embeddings_size 300 --use_structured_data --structured_data_input_dim 3
    python3 -m syn.model.build.codebooks.evaluate --task prioritization --corpus openOffice --embeddings_model glove --embeddings_size 300 --use_structured_data --structured_data_input_dim 3
    python3 -m syn.model.build.codebooks.search_hyperparameters --reference_params --params_file syn/model/build/codebooks/parameter_space/openOffice/prioritization/params_space_pretrained_use_structured_data_decision_tree.json &
~~~


[msr_repository]: http://alazar.people.ysu.edu/msr14data/
[bugzilla_api]: https://bugzilla.readthedocs.io/en/latest/api/
[gerrit_api]: https://gerrit-review.googlesource.com/Documentation/rest-api.html

