/** @type {import('next').NextConfig} */
const nextConfig = {
  // Don't expose the Next.js version header to clients
  poweredByHeader: false,

  // Enable gzip/brotli compression at the Next.js layer
  compress: true,

  // Strict mode surfaces extra React warnings in development
  reactStrictMode: true,

  // Security response headers applied to every route
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          // Prevent the page from being embedded in iframes (clickjacking)
          { key: "X-Frame-Options", value: "DENY" },
          // Block MIME-type sniffing
          { key: "X-Content-Type-Options", value: "nosniff" },
          // Only send the origin as referrer, never the full URL
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          // Restrict powerful browser features
          {
            key: "Permissions-Policy",
            value: "camera=(), microphone=(), geolocation=()",
          },
          // Basic CSP: tighten further once you know all asset domains
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline'",   // Next.js requires unsafe-inline for hydration
              "style-src 'self' 'unsafe-inline'",
              "img-src 'self' data: blob:",
              "font-src 'self'",
              "connect-src 'self' https:",            // allows fetch() to your API over HTTPS
              "frame-ancestors 'none'",
            ].join("; "),
          },
        ],
      },
    ];
  },
};

export default nextConfig;
