import React from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { BookOpen, PenTool, Mic, Eye } from 'lucide-react';

export default function Home() {
  return (
    <>
      <Head>
        <title>TOEFL/IELTS AI Teaching Platform</title>
        <meta name="description" content="Master TOEFL and IELTS with AI-powered personalized learning" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        {/* Header */}
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-6">
              <div className="flex items-center">
                <BookOpen className="h-8 w-8 text-blue-600" />
                <h1 className="ml-2 text-2xl font-bold text-gray-900">
                  TOEFL/IELTS AI Tutor
                </h1>
              </div>
              <nav className="hidden md:flex space-x-8">
                <Link href="/grammar" className="text-gray-500 hover:text-gray-900">
                  Grammar
                </Link>
                <Link href="/reading" className="text-gray-500 hover:text-gray-900">
                  Reading
                </Link>
                <Link href="/writing" className="text-gray-500 hover:text-gray-900">
                  Writing
                </Link>
                <Link href="/speaking" className="text-gray-500 hover:text-gray-900">
                  Speaking
                </Link>
              </nav>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
              <h2 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
                Master TOEFL & IELTS with
                <span className="text-blue-600"> AI-Powered Learning</span>
              </h2>
              <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
                Get personalized feedback, practice with real exam questions, and improve your English skills 
                with our advanced AI tutoring system.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link
                  href="/grammar"
                  className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
                >
                  Start Learning
                </Link>
                <Link
                  href="/about"
                  className="border border-gray-300 text-gray-700 px-8 py-3 rounded-lg font-semibold hover:bg-gray-50 transition-colors"
                >
                  Learn More
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h3 className="text-3xl font-bold text-gray-900 mb-4">
                Master All Four Skills
              </h3>
              <p className="text-lg text-gray-600">
                Comprehensive preparation for TOEFL and IELTS exams
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
              {/* Grammar */}
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-8 rounded-xl">
                <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center mb-6">
                  <PenTool className="h-6 w-6 text-white" />
                </div>
                <h4 className="text-xl font-semibold text-gray-900 mb-4">Grammar</h4>
                <p className="text-gray-600 mb-6">
                  Practice grammar exercises with instant feedback and detailed explanations.
                </p>
                <Link
                  href="/grammar"
                  className="text-blue-600 font-semibold hover:text-blue-700"
                >
                  Start Grammar Practice →
                </Link>
              </div>

              {/* Reading */}
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-8 rounded-xl">
                <div className="w-12 h-12 bg-green-600 rounded-lg flex items-center justify-center mb-6">
                  <Eye className="h-6 w-6 text-white" />
                </div>
                <h4 className="text-xl font-semibold text-gray-900 mb-4">Reading</h4>
                <p className="text-gray-600 mb-6">
                  Improve comprehension with practice passages and detailed answer explanations.
                </p>
                <Link
                  href="/reading"
                  className="text-green-600 font-semibold hover:text-green-700"
                >
                  Start Reading Practice →
                </Link>
              </div>

              {/* Writing */}
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-8 rounded-xl">
                <div className="w-12 h-12 bg-purple-600 rounded-lg flex items-center justify-center mb-6">
                  <BookOpen className="h-6 w-6 text-white" />
                </div>
                <h4 className="text-xl font-semibold text-gray-900 mb-4">Writing</h4>
                <p className="text-gray-600 mb-6">
                  Get AI-powered essay scoring and personalized feedback for improvement.
                </p>
                <Link
                  href="/writing"
                  className="text-purple-600 font-semibold hover:text-purple-700"
                >
                  Start Writing Practice →
                </Link>
              </div>

              {/* Speaking */}
              <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-8 rounded-xl">
                <div className="w-12 h-12 bg-orange-600 rounded-lg flex items-center justify-center mb-6">
                  <Mic className="h-6 w-6 text-white" />
                </div>
                <h4 className="text-xl font-semibold text-gray-900 mb-4">Speaking</h4>
                <p className="text-gray-600 mb-6">
                  Practice speaking with AI assessment of pronunciation and fluency.
                </p>
                <Link
                  href="/speaking"
                  className="text-orange-600 font-semibold hover:text-orange-700"
                >
                  Start Speaking Practice →
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20 bg-blue-600">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h3 className="text-3xl font-bold text-white mb-4">
              Ready to Ace Your TOEFL or IELTS?
            </h3>
            <p className="text-xl text-blue-100 mb-8">
              Join thousands of students who have improved their scores with our AI tutor.
            </p>
            <Link
              href="/grammar"
              className="bg-white text-blue-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              Get Started Today
            </Link>
          </div>
        </section>

        {/* Footer */}
        <footer className="bg-gray-900 text-white py-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
              <div className="flex items-center justify-center mb-4">
                <BookOpen className="h-8 w-8 text-blue-400" />
                <h4 className="ml-2 text-xl font-bold">TOEFL/IELTS AI Tutor</h4>
              </div>
              <p className="text-gray-400">
                © 2024 TOEFL/IELTS AI Teaching Platform. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </main>
    </>
  );
}