#!/bin/zsh
PROJECT_ID="aa-2019"
BUCKET_NAME="aa-2019-mlengine"
REGION=us-central1

# para crear : gsutil mb -l $REGION gs://$BUCKET_NAME
#gsutil cp -r data gs://$BUCKET_NAME/data
#TRAIN_DATA=gs://$BUCKET_NAME/data/adult.data.csv
#EVAL_DATA=gs://$BUCKET_NAME/data/adult.test.csv
#gsutil cp ../test.json gs://$BUCKET_NAME/data/test.json
#TEST_JSON=gs://$BUCKET_NAME/data/test.json

JOB_NAME=mnist-1 #tiene que ir cambiando
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.10 \
    --python-version 3.5 \
    --module-name trainer.model \
    --package-path trainer/ \
    --region $REGION \

    

gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir $MODEL_DIR \

