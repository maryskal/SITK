#!/bin/bash

# Parametros de entrada, cid = nombre del ct, path = carpeta para compartir
# bash cip_vascular_analysis.sh nombre_tc carpeta_compartida 
cid=$1
path=$2



	# Ejecuta el contendor, crea el volumen con path, y lo conecta con host
	cont=`docker run -ti -v ${path}:/host -d acilbwh/chestimagingplatform:bdc`

	# Crea una variable para no ejecutar todo el rato todo ese texto
	export dock="docker exec -ti ${cont} "

	# Copias ct a carpeta temporal (por si hay errores)
	$dock cp /host/${cid}.nrrd /tmp/${cid}.nrrd

	# DOWNSAMPLING
	# Parametros para el downsampling
	z_size=`$dock unu head ${cid}.nrrd | grep sizes | cut -d" " -f4`
	spacing=`$dock unu head ${cid}.nrrd | grep directions | cut -d, -f7 | cut -d\) -f1`
	down=`awk "BEGIN {print ${spacing}/2}"`
	up=`awk "BEGIN {print 2/${spacing}}"`

	# SEGMENTACION (si ya esta hecha no se hace)

		# Funcion downsampling
		$dock unu resample -k tent -s = = x${down} -i /tmp/${cid}.nrrd -c cell -o /tmp/${cid}_resampled.nrrd

		# Segmentacion de pulmon
		$dock python /ChestImagingPlatform/cip_python/dcnn/projects/lung_segmenter/lung_segmenter_dcnn.py \
		--i /tmp/${cid}_resampled.nrrd --t combined  --o /tmp/${cid}_partialLungLabelMap_resampled.nrrd

		# Resampling (solo se hace downsampling para la segmentacion)
		$dock unu resample -k cheap -s = = ${z_size} -i /tmp/${cid}_partialLungLabelMap_resampled.nrrd  -c cell -o /tmp/${cid}_partialLungLabelMap.nrrd
		# Comprimir el output
		$dock unu save -f nrrd -e gzip -i /tmp/${cid}_partialLungLabelMap.nrrd -o /tmp/${cid}_partialLungLabelMap.nrrd
		# Lo guardas en host
		$dock cp /tmp/${cid}_partialLungLabelMap.nrrd /host/${cid}_partialLungLabelMap.nrrd



	# SACAR LOS VASOS

		# Sacar las particulas de vasos (pulmon derecho e izquierdo)
		$dock python /ChestImagingPlatform/Scripts/cip_compute_vessel_particles.py --tmpDir /tmp  -i /tmp/${cid}.nrrd -l /tmp/${cid}_partialLungLabelMap.nrrd -r RightLung --init Frangi \
		-o /tmp/${cid} --liveTh -120 --seedTh -80 -s 0.625 --cleanCache
		$dock python /ChestImagingPlatform/Scripts/cip_compute_vessel_particles.py --tmpDir /tmp  -i /tmp/${cid}.nrrd -l /tmp/${cid}_partialLungLabelMap.nrrd -r LeftLung --init Frangi \
		-o /tmp/${cid} --liveTh -120 --seedTh -80 -s 0.625 --cleanCache

		# Juntar los dos pulmones
		$dock MergeParticleDataSets -i /tmp/${cid}_rightLungVesselParticles.vtk -i /tmp/${cid}_leftLungVesselParticles.vtk --out /tmp/${cid}_wholeLungVesselParticles.vtk

		# Copiar el resultado a la carpeta compartida
		$dock cp /tmp/${cid}_wholeLungVesselParticles.vtk /host/${cid}_wholeLungVesselParticles.vtk



	# VASCULAR PHENOTYPES (volumen vascular por tamanio de vaso)

		pairs="WholeLung,Vessel,RightLung,Vessel,LeftLung,Vessel"
		# Etiqueta los vasos segun su tamanio
		$dock LabelParticlesByChestRegionChestType --ilm /tmp/${cid}_partialLungLabelMap.nrrd  --ip /tmp/${cid}_wholeLungVesselParticles.vtk --op /tmp/${cid}_vesselPartial.vtk
		# Crea un csv con el volumen por tamanio de vaso
		$dock python /ChestImagingPlatform/cip_python/phenotypes/vasculature_phenotypes.py -i /tmp/${cid}_vesselPartial.vtk --cid ${cid} -p ${pairs}  --out_csv /tmp/${cid}_vascularPhenotypes.csv
		# Copiar el resultado a la carpeta compartida
		$dock cp /tmp/${cid}_vascularPhenotypes.csv /host/${cid}_vascularPhenotypes.csv



	docker stop ${cont}
	docker rm ${cont}