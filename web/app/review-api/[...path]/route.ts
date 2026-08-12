import { LocalProxyError, proxyLocalReview } from "@/lib/review-proxy";

type RouteContext = { params: Promise<{ path: string[] }> };

export async function GET(request: Request, context: RouteContext): Promise<Response> {
  return handle(request, context, "GET");
}

export async function POST(request: Request, context: RouteContext): Promise<Response> {
  return handle(request, context, "POST");
}

async function handle(
  request: Request,
  context: RouteContext,
  method: "GET" | "POST"
): Promise<Response> {
  try {
    const { path } = await context.params;
    return await proxyLocalReview(request, path, method);
  } catch (reason) {
    if (reason instanceof LocalProxyError) {
      return Response.json({ detail: reason.message }, { status: 400 });
    }
    return Response.json({ detail: "The local review adapter is unavailable." }, { status: 502 });
  }
}
