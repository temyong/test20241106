gcloud config set project englishcentral-qa-428220
gcloud run jobs deploy vocab-checker \
    --source . \
    --tasks 1 \
    --set-env-vars MODEL_VENDOR=openai \
    --region us-east1
    --project=englishcentral-qa-428220
gcloud run jobs execute vocab-checker --region us-east1
