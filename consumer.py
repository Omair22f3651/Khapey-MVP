import pika
import json
import requests
import io
import logging
import os
from dotenv import load_dotenv
import base64
from app import KhapeyMVP  # Assume this is the file with KhapeyMVP class

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Initialize KhapeyMVP
mvp = KhapeyMVP()

def callback(ch, method, properties, body):
  try:
    payload = json.loads(body)
    review_id = payload.get('reviewId')
    user_id = payload.get('userId')
    user_email = payload.get('userEmail', '')
    review_data = payload.get('reviewData', {})
    media_urls = review_data.get('mediaUrls', [])

    logging.info(f"Received review event: reviewId={review_id}, userId={user_id}, email={user_email}")

    # Download media files
    media_files = []
    for media in media_urls:
      try:
        response = requests.get(media['url'], stream=True, timeout=30)
        response.raise_for_status()
        file_stream = io.BytesIO(response.content)
        file_stream.filename = media['url'].split('/')[-1]
        file_stream.content_type = media['mediaType']
        media_files.append(file_stream)
        logging.info(f"Downloaded media: {media['url']}")
      except requests.RequestException as e:
        logging.error(f"Failed to download media {media['url']}: {str(e)}")
        continue

    if not media_files:
      logging.warning(f"No media files downloaded for review {review_id}")
      ch.basic_ack(delivery_tag=method.delivery_tag)
      return

    # Analyze review
    results = mvp.analyze_review(media_files, review_data)
    logging.info(f"Analysis complete for review {review_id}: quality_score={results['quality_score']}")

    # Store in Qdrant with mongodb_id
    image_embeddings = [mvp._get_media_embedding(f, f.content_type.startswith('video/')) for f in media_files]
    image_base64_list = [mvp._get_video_keyframe_base64(f) if f.content_type.startswith('video/') else base64.b64encode(f.read()).decode('utf-8') for f in media_files]
    mvp._store_review_in_qdrant(results['ai_analysis'], {**review_data, 'mongodb_id': review_id, 'user_id': user_id, 'user_email': user_email}, image_embeddings, image_base64_list)

    logging.info(f"Stored analysis in Qdrant for review {review_id}")
    ch.basic_ack(delivery_tag=method.delivery_tag)
  except Exception as e:
    logging.error(f"Error processing review {review_id}: {str(e)}", exc_info=True)
    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host="localhost",
        port=5672,
        credentials=pika.PlainCredentials("guest", "guest")
    )
)


channel = connection.channel()
channel.queue_declare(queue='review_analysis', durable=True)
channel.basic_consume(queue='review_analysis', on_message_callback=callback)
logging.info('Waiting for review events...')
channel.start_consuming()