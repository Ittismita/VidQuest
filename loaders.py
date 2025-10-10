from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api.formatters import TextFormatter

def fetch_transcript(video_id: str) -> str:
    try:
        ytt_api = YouTubeTranscriptApi()
        formatter = TextFormatter()
        transcript_list = ytt_api.fetch(video_id=video_id, languages=["en"])
        return formatter.format_transcript(transcript_list)
    except TranscriptsDisabled:
        return ""
