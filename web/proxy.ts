import { NextRequest, NextResponse } from "next/server";

import { configuredWebPort, LocalProxyError, validateLocalWebRequest } from "@/lib/review-proxy";

export function proxy(request: NextRequest): NextResponse {
  try {
    validateLocalWebRequest(request, configuredWebPort());
  } catch (error) {
    if (error instanceof LocalProxyError) {
      return NextResponse.json({ detail: error.message }, { status: 400 });
    }
    return NextResponse.json({ detail: "Local review request validation failed." }, { status: 400 });
  }
  return NextResponse.next();
}

export const config = {
  matcher: "/:path*"
};
