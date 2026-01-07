export async function GET(req, { params }) {
  try {
    const { videoUrl } = params;
    const decodedUrl = decodeURIComponent(videoUrl);

    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/progress/advanced/${encodeURIComponent(decodedUrl)}`);

    if (!response.ok) {
      const errorData = await response.json();
      return new Response(
        JSON.stringify({ error: errorData.detail || 'Failed to get progress' }),
        { status: response.status, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const data = await response.json();
    return new Response(
      JSON.stringify(data),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
