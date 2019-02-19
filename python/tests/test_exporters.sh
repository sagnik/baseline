#!/usr/bin/env bash

function tf_version_test {
    TF_VERSION_TEST=`python -c 'import tensorflow; from distutils.version import LooseVersion; import sys; i = "fail" if LooseVersion(tensorflow.__version__) < LooseVersion("1.12.0") else "pass"; print(i)'`

    if [ "$TF_VERSION_TEST" == "fail" ]
        then
            printf "${ERROR_COLOR}models were trained with tf version 1.12, you have a lower version. please upgrade.\n${END}"
            exit 1
    fi
}

function docker_clear {
    docker stop ${SERVING_CONTAINER_NAME} > /dev/null 2>&1
    docker rm ${SERVING_CONTAINER_NAME} > /dev/null 2>&1
}

function docker_run {
    docker run -p ${REMOTE_PORT_HTTP}:${REMOTE_PORT_HTTP} -p ${REMOTE_PORT_GRPC}:${REMOTE_PORT_GRPC} --name ${SERVING_CONTAINER_NAME} -v $1 -e MODEL_NAME=${MODEL_NAME} -t tensorflow/serving &
}

function get_file {
     if [ -f $1 ];
         then
         printf "${MSG_COLOR} $1 locally found, not downloading\n${END}"
     else
         printf "${MSG_COLOR} $1 locally not found, downloading $2\n${END}"
         wget $2 -O $1
     fi
}

function mead_export {
    mead-export --config ${CONFIG_FILE} --settings ${EXPORT_SETTINGS_MEAD} --datasets ${EXPORT_SETTINGS_DATASETS} --task ${TASK} --exporter_type ${EXPORTER_TYPE} --model ${MODEL_FILE} --model_version ${MODEL_VERSION} --output_dir $1 --is_remote ${IS_REMOTE}
}

function check_diff {
    DIFF=$(diff ${1} ${2})
        if [ "$DIFF" != "" ]
        then
            printf "${ERROR_COLOR}${1} does not match with ${2}, exporting failed. \n${END}"
            exit 1
        fi
}

function remove_files {
    arr=("$@")
    for file in "${arr[@]}"
        do
            [ -e "${file}" ] && rm -rf "${file}"
        done
}

function classify_text {
    if [ -z "$2" ]
    then
        python ${DRIVER} --model $1 --text ${TEST_FILE} --backend tf -name ${MODEL_NAME} --preproc $3 > $4
    else
        printf "${MSG_COLOR} ${1} ${2} ${3} ${4}${END}"
        python ${DRIVER} --model $1 --text ${TEST_FILE} --backend tf --remote ${2} --name ${MODEL_NAME} --preproc $3 > $4
    fi
}


## get the variables defined in the config into shell
eval $(sed -e 's/:[^:\/\/]/="/g;s/$/"/g;s/ *=/=/g' $1)

## check tf version
tf_version_test
docker_clear

## remove files from previous run, if any
FILES_TO_REMOVE=(${TEST_LOAD} ${TEST_SERVE} ${TEST_SERVE_PREPROC} ${EXPORT_DIR} ${EXPORT_DIR_PREPROC})
remove_files "${FILES_TO_REMOVE[@]}"

## how many lines coming from baseline prints need to be removed?
NUM_LINES_TO_REMOVE_LOAD=`expr "$NUM_FEATURES" + 3`
NUM_LINES_TO_REMOVE_SERVE=`expr "$NUM_FEATURES" + 2`

printf "${MSG_COLOR}running test for ${TASK}\n${END}"
printf "${MSG_COLOR}------------------------\n${END}"

## get the files
get_file ${MODEL_FILE} ${MODEL_FILE_LINK}
get_file ${TEST_FILE} ${TEST_FILE_LINK}
get_file ${CONFIG_FILE} ${CONFIG_FILE_LINK}

### load model and classify
#printf "${MSG_COLOR}processing by loading the model\n${END}"
#classify_text ${MODEL_FILE} "" client ${TEST_LOAD}  # remote end points are empty, preproc is client
#sleep ${SLEEP}

## export with preproc=client and classify
#printf "${MSG_COLOR}exporting model with preproc=client\n${END}"
#mkdir -p ${EXPORT_DIR}
#mead_export ${EXPORT_DIR}/${MODEL_NAME}
#sleep ${SLEEP}
#printf "${MSG_COLOR}running tf serving\n${END}"
#docker_clear
#docker_run ${EXPORT_DIR}:/models
#sleep ${SLEEP}
#printf "${MSG_COLOR}processing with served model, preproc=client\n${END}"
#classify_text ${EXPORT_DIR}/${MODEL_NAME}/1/ ${REMOTE_HOST}:${REMOTE_PORT_GRPC} client ${TEST_SERVE} # valid remote end points, preproc is client.
#sleep ${SLEEP}
## remove first few lines and check if the outputs match
#sed -i -e 1,${NUM_LINES_TO_REMOVE_LOAD}d ${TEST_LOAD}
#sed -i -e 1,${NUM_LINES_TO_REMOVE_SERVE}d ${TEST_SERVE}
#check_diff ${TEST_LOAD} ${TEST_SERVE}

## export with preproc=server and classify
printf "${MSG_COLOR}exporting model with preproc=server\n${END}"
mkdir -p ${EXPORT_DIR_PREPROC}
mead_export ${EXPORT_DIR_PREPROC}/${MODEL_NAME}
sleep ${SLEEP}
printf "${MSG_COLOR}running tf serving\n${END}"
docker_clear
docker_run ${EXPORT_DIR_PREPROC}:/models
printf "${MSG_COLOR}processing with served model, preproc=server\n${END}"
classify_text ${EXPORT_DIR_PREPROC}/${MODEL_NAME}/1/ ${REMOTE_HOST}:${REMOTE_PORT_GRPC} server ${TEST_SERVE_PREPROC} # valid remote end points, preproc is server.
docker_clear
# remove first few lines and check if the outputs match
sed -i -e 1,${NUM_LINES_TO_REMOVE_SERVE}d ${TEST_SERVE_PREPROC}
check_diff ${TEST_SERVE} ${TEST_SERVE_PREPROC}
printf "${MSG_COLOR}${TASK} export successful.\n${END}"

## if successful, clean the files
if [ "$CLEAN_AFTER_TEST" == "true" ]
then
    mead-clean
    FILES_TO_REMOVE=( ${MODEL_FILE}, ${TEST_FILE}, ${CONFIG_FILE}, ${EXPORT_DIR}, ${EXPORT_DIR_PREPROC}, ${TEST_LOAD}, ${TEST_SERVE}, ${TEST_SERVE_PREPROC})
    remove_files ${FILES_TO_REMOVE}
fi
exit 0
