import type { NextConfig } from "next";

const reviewApiUrl = process.env.WMO_REVIEW_API_URL ?? "http://127.0.0.1:8017";
const reviewApiHost = new URL(reviewApiUrl).hostname;

if (!["127.0.0.1", "::1", "localhost"].includes(reviewApiHost)) {
  throw new Error("WMO_REVIEW_API_URL must point to a loopback local review adapter.");
}

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/review-api/:path*",
        destination: `${reviewApiUrl}/:path*`
      }
    ];
  }
};

export default nextConfig;
