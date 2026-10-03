"""Radiogenómica: relacionar lo que se mide en la RM con la ómica del mismo paciente.

Este subpaquete implementa el plan de ``docs/PLAN_ANALISIS_RADIOGENOMICA.md``:

    imaging.py     volúmenes y fracción necrótica desde máscaras BraTS
    cohorts.py     identificadores de paciente, solapamiento y potencia, descargas
                   (cBioPortal para TCGA, paquete ``cptac`` para CPTAC)
    signatures.py  lectura de firmas GMT y puntuación por muestra
    stats.py       correlación parcial de Spearman, permutación, bootstrap,
                   control con firmas aleatorias, FDR
    analysis.py    orquesta los análisis confirmatorio, de validación y exploratorio
    synthetic.py   cohorte multiómica sintética con efecto conocido

Idea central: todas las tablas son DataFrames con **pacientes en el índice**
(``patient_id`` normalizado). Unir imagen y ómica es entonces un ``join`` por
índice, y el número de pacientes en común se ve en cada paso.
"""
