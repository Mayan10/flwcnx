import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/theme-provider";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FlowConX — Network Intelligence Platform",
  description:
    "Real-time network flow intelligence, classification & observability. FlowConX processes network and satellite traffic data using ML-powered classification.",
  keywords: [
    "network intelligence",
    "traffic classification",
    "flow analysis",
    "observability",
    "satellite connectivity",
    "ML classification",
  ],
  openGraph: {
    title: "FlowConX — Network Intelligence Platform",
    description:
      "Real-time network flow intelligence, classification & observability.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      data-theme="dark"
      className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      suppressHydrationWarning
    >
      <body>
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
