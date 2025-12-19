import React, { useState, useRef, useEffect } from 'react';
import { 
  AlertCircle, CheckCircle, Upload, Link, Type, Loader2, 
  AlertTriangle, FileText, ShieldCheck, Sparkles, Github, 
  Cpu, Database, Code, Zap, Brain, Layout, Server, GitBranch, ArrowRight, ExternalLink, Globe, Lightbulb, HelpCircle, Copy, X, CornerUpRight, Terminal, BookOpen, Bug
} from 'lucide-react'; 

// !!! LOCAL TESTING MODE !!!
// Pointing to your local Python backend. 
const API_BASE_URL = "http://localhost:8000";

// --- TECHNOLOGY DATA ---
const TECHNOLOGIES = [
    { name: "Python", icon: Terminal, link: "https://www.python.org/" },
    { name: "React", icon: Zap, link: "https://reactjs.org/" },
    { name: "Hugging Face", icon: GitBranch, link: "https://huggingface.co/" },
    { name: "Kaggle Datasets", icon: Database, link: "https://www.kaggle.com/datasets" },
    { name: "Gemini AI Studio", icon: Sparkles, link: "https://ai.google.dev/" },
];

// --- STEP CONTENT MAPPING ---
const STEP_CONTENT = {
    'Input': { 
        icon: Upload,
        title: "Step 1: Input & Pre-processing", 
        detail: "The analyzer accepts raw text, a URL (where it uses the domain for initial context), or a document (PDF/Image). For documents, the system first runs an extraction check: Digital PDFs use direct text extraction for high accuracy, while scanned images/PDFs are converted to high-DPI images and analyzed via local OCR (PaddleOCR or TrOCR fallback). This ensures clean, structured input for the next stage." 
    },
    'Decomposition (LLM/VLM)': { 
        icon: Brain,
        title: "Step 2: Claim Decomposition & FKC", 
        detail: "The input text is broken down into atomic claims. A Linguistic Neural Network (DistilBERT) annotates each claim with four signals. \n\nAdditionally, the **Fundamental Knowledge Checker (FKC)** scans claims against physical and biological impossibilities. It acts as a guardrail for violations of physical laws (e.g., 'gravity falls up') or biological necessities, leaving symbolic or definitional nuances to the consensus layer." 
    },
    'Annotation (NN + LLM)': { 
        icon: Database,
        title: "Step 3: Signal Scoring (The NN Engine)", 
        detail: "Each individual claim is passed through a specialized, multi-head DistilBERT Neural Network (NN). This NN runs four independent deep-context predictions (the 'signals'): Plausibility, Evidence Specificity, Bias Level, and Uncertainty Presence. These scores (0.0 to 1.0) provide the granular data necessary for the final decision-making layer, using tunable thresholds (e.g., 0.8 for High, 0.6 for Medium) for categorization." 
    },
    'Judgment (Meta-NN)': { 
        icon: ShieldCheck,
        title: "Step 4: Final Verdict: Confidence-Aware Neural Judge", 
        detail: "This final phase utilizes the **Hierarchical Claim-Level Decision Model (HCDM)**. It aggregates signals to produce a credibility score. Note that the **Knowledge Sanity Assistant (KSA)**, previously used for heuristic patterns, has been **disabled** for this version to rely purely on neural and HCDM signals."
    }
};

// Define special ID for About Page transition
const ABOUT_ID = 'AboutPage';

// --- ABOUT PAGE COMPONENT ---
// Modified AboutPage to accept a closer function (closePage) instead of navigateTo
const AboutPage = ({ closePage }) => (
    // Note: The outer div padding is removed as this component is rendered inside the ExpandingStepDetail fixed container
    <div className="w-full h-full">
        <div className="pt-12 pb-20 max-w-3xl mx-auto px-6">
            {/* The Back button here handles the transition reversal */}
            <button
                onClick={closePage}
                className="flex items-center gap-2 text-indigo-600 hover:text-indigo-800 font-semibold mb-6 transition-colors"
            >
                <ArrowRight className="w-4 h-4 transform rotate-180"/>
                Back to Analyzer
            </button>
            <div className="bg-white/90 backdrop-blur-md rounded-3xl shadow-xl border border-gray-100 p-8 md:p-12 space-y-8">
                <div className="text-center">
                    <h1 className="text-4xl font-black text-indigo-700 mb-2 tracking-tight">
                        About Claim Analyzer
                    </h1>
                    <p className="text-gray-500 text-lg font-medium">A Hybrid Approach to Credibility Scoring</p>
                </div>

                {/* Original Back Button is removed to avoid conflict with animated one */}
                
                <section className="space-y-4 pt-4 border-t border-gray-100">
                    <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                        <Sparkles className="w-6 h-6 text-indigo-500"/>
                        Motivation & Goal
                    </h3>
                    <p className="text-gray-700 leading-relaxed">
                        Thank you for exploring our project. We developed the Claim Analyzer to address the growing challenge of information overload and sophisticated digital deception. Simple binary (True/False) systems are often inadequate for modern misinformation, which frequently relies on biased framing or vague sourcing. Our goal was to create a system that provides granular, explainable signals—allowing users to understand *why* a claim might be untrustworthy, not just *whether* it is true.
                    </p>
                </section>

                <section className="space-y-4">
                    <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                        <Layout className="w-6 h-6 text-cyan-500"/>
                        Approach: The Hybrid Signal Model
                    </h3>
                    <p className="text-gray-700 leading-relaxed">
                        We utilized a Hybrid Signal Model approach, combining the predictive power of a fine-tuned Neural Network (NN) with advanced decision structures.
                    </p>
                    <ul className="list-disc list-inside text-gray-700 ml-4 space-y-2 text-sm">
                        <li>VLM/OCR: Used for robust text extraction from Documents and Images.</li>
                        <li>Specialized NN (DistilBERT): A locally trained, multi-head transformer model used for *Deep Signal Scoring* (Plausibility, Evidence, Bias, Uncertainty).</li>
                        <li>**Hierarchical Claim-Level Decision Model (HCDM):** This intermediate layer classifies individual claims (Supported, Contradicted, Speculative) to provide robust aggregate metrics.</li>
                        <li>**Meta-NN Judge:** Replaces rigid rules with a trained Logistic Regression model to synthesize the final verdict based on learned non-linear relationships between all signals.</li>
                    </ul>
                </section>

                <section className="space-y-4">
                    <h3 className="2xl font-bold text-gray-800 flex items-center gap-2">
                        <Code className="w-6 h-6 text-emerald-500"/>
                        Technologies Utilized
                    </h3>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                        {[
                            { tech: "Python/FastAPI", desc: "Backend Server" },
                            { tech: "React/Tailwind", desc: "Responsive Frontend" },
                            { tech: "PyTorch/Transformers", desc: "NN Training Framework" },
                            { tech: "Scikit-learn", desc: "Meta-NN Training" },
                            { tech: "PyMuPDF", desc: "PDF Handling" },
                            { tech: "PaddleOCR", desc: "OCR Model" },
                        ].map((item) => (
                            <div key={item.tech} className="bg-gray-50 p-3 rounded-lg border border-gray-100 shadow-sm">
                                <h4 className="font-semibold text-gray-900">{item.tech}</h4>
                                <p className="text-xs text-gray-500">{item.desc}</p>
                            </div>
                        ))}
                    </div>
                </section>
            </div>
        </div>
    </div>
);
// --- END ABOUT PAGE COMPONENT ---

// --- STEP DETAILS PAGE COMPONENT (REFACTORED FOR ANIMATION) ---
const ExpandingStepDetail = ({ transitionDetails, handleBackTransition, isActive }) => {
    // Check if we are rendering the general About page or a specific step
    const isAboutPage = transitionDetails.id === ABOUT_ID;
    const step = isAboutPage ? null : STEP_CONTENT[transitionDetails.id];
    
    // Icon component for steps
    const IconComponent = isAboutPage ? Sparkles : step.icon;
    
    // Content for steps (if not About page)
    const formattedDetail = step ? step.detail.split('\n').map((paragraph, index) => (
        <p key={index} className={`mb-4 text-gray-700 leading-relaxed whitespace-pre-wrap text-base md:text-lg`}>
            {paragraph}
        </p>
    )) : null;

    // --- Styles for Smooth Transition (Expansion and Fade Out) ---
    const rect = transitionDetails.rect;
    
    // Determine target size/position based on isActive state
    const targetLeft = isActive ? 0 : rect.left;
    const targetTop = isActive ? 0 : rect.top;
    const targetWidth = isActive ? window.innerWidth : rect.width;
    const targetHeight = isActive ? window.innerHeight : rect.height;
    const targetBorderRadius = isActive ? '0rem' : '1rem';
    const targetOpacity = isActive ? 1 : 0; 
    
    const expandingCardStyle = {
        position: 'fixed',
        left: targetLeft,
        top: targetTop,
        width: targetWidth,
        height: targetHeight,
        
        transition: 'all 500ms cubic-bezier(0.4, 0.0, 0.2, 1)', 
        
        zIndex: 1000,
        backgroundColor: 'white',
        borderRadius: targetBorderRadius, 
        boxShadow: isActive ? '0 0 50px rgba(0,0,0,0.5)' : '0 15px 30px rgba(0,0,0,0.2)',
        overflowY: isActive ? 'auto' : 'hidden', 
        overflowX: 'hidden',
        opacity: targetOpacity, 
    };
    
    const [isContentVisible, setIsContentVisible] = useState(false);
    
    useEffect(() => {
        if (isActive) {
            const timer = setTimeout(() => setIsContentVisible(true), 550); 
            return () => clearTimeout(timer);
        } else {
            setIsContentVisible(false);
        }
    }, [isActive]);

    // Style for content reveal (Phase 4: Slide up from bottom)
    const contentRevealClass = isContentVisible 
        ? 'opacity-100 translate-y-0'
        : 'opacity-0 translate-y-8';
    
    return (
        <div style={expandingCardStyle} className="pointer-events-auto">
            
            <div className={`
                w-full h-full transition-opacity duration-300 ease-out 
                ${isContentVisible ? 'opacity-100' : 'opacity-0'}
            `}>
                {isAboutPage ? (
                    // Renders the full AboutPage component inside the fixed transition container
                    <AboutPage closePage={handleBackTransition} /> 
                ) : (
                    // Renders the individual Step Detail content
                    <div className="max-w-3xl mx-auto p-6 md:p-12">
                         
                        <button
                            onClick={handleBackTransition}
                            className={`flex items-center gap-2 text-indigo-600 hover:text-indigo-800 font-semibold mb-6 transition-all duration-300 transform ${contentRevealClass} delay-100`}
                        >
                            <ArrowRight className="w-4 h-4 transform rotate-180"/>
                            Back to Analyzer
                        </button>

                        <div className="text-center space-y-4 pt-6">
                            <div className={`w-16 h-16 mx-auto bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600 transition-all transform ${contentRevealClass} delay-200`}>
                                <IconComponent className="w-8 h-8" />
                            </div>
                            <h1 className={`text-4xl font-black text-gray-900 tracking-tight transition-all transform ${contentRevealClass} delay-300`}>
                                {step.title}
                            </h1>
                        </div>

                        <section className={`mt-8 pt-6 border-t border-gray-100 transition-all transform ${contentRevealClass} delay-400`}>
                            {formattedDetail}
                        </section>
                    </div>
                )}
            </div>
        </div>
    );
};


export default function App() {
  const [currentPage, setCurrentPage] = useState('home'); // 'home', 'about', or 'details'
  const [selectedStepId, setSelectedStepId] = useState(null); // Key for the step content
  const [transitionActive, setTransitionActive] = useState(false); // Manages Phase 2 animation (Expansion)
  const [transitionDetails, setTransitionDetails] = useState(null); // Stores rect and id for transition
  const [focusedCardId, setFocusedCardId] = useState(null); // Manages Phase 1 animation (Focus)
  const cardRefs = useRef({}); // To store refs for each "How It Works" card
  const aboutButtonRef = useRef(null); // NEW REF for About button

  // Renamed 'image' tab to 'document'
  const [activeTab, setActiveTab] = useState('text'); // 'text', 'url', 'document'
  const [inputText, setInputText] = useState('');
  const [inputUrl, setInputUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [debugMode, setDebugMode] = useState(false); // New: Debug Mode State
  
  const [copiedStatus, setCopiedStatus] = useState(false);
  const [isStyleReady, setIsStyleReady] = useState(true); 
  const howItWorksRef = useRef(null);
  const fileInputRef = useRef(null); // Ref for file input element
  
  // Removed manual crop state and dependency
  // const [manualCropCoords, setManualCropCoords] = useState(''); // "x1,y1,x2,y2"
  
  // !!! LOCAL TESTING MODE !!!
  const API_BASE_URL = "http://localhost:8000";


  useEffect(() => {
    document.title = "FactCheck AI - Signal Analyzer";
  }, []);
  
  // Custom Hook to manage body overflow based on transition state
  useEffect(() => {
    // Only lock scroll when actively transitioning to the details view (Phase 2)
    if (transitionActive && currentPage !== 'details') {
      document.body.style.overflow = 'hidden';
    } else if (!transitionActive && transitionDetails) {
        // If transition is inactive and details are present (meaning we just finished shrinking back)
        // This handles restoring scroll after the back transition completes
        const timer = setTimeout(() => {
            document.body.style.overflow = 'auto';
            window.scrollTo({ top: transitionDetails.originalScrollTop, behavior: 'instant' });
            setTransitionDetails(null); // Clean up details state after restoration
            setFocusedCardId(null); // CRITICAL FIX: Clear blur/focus state
        }, 500); // Wait for the reverse animation (500ms)
        return () => clearTimeout(timer);
    } else {
         document.body.style.overflow = 'auto';
    }
  }, [transitionActive, currentPage, transitionDetails]);


  // --- QoL Feature: Copy to Clipboard Function ---
  const copyToClipboard = (textToCopy) => {
    if (!textToCopy) return;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(textToCopy).then(() => {
        setCopiedStatus(true);
        setTimeout(() => setCopiedStatus(false), 2000);
      }).catch(err => {
        fallbackCopyTextToClipboard(textToCopy);
      });
    } else {
      fallbackCopyTextToClipboard(textToCopy);
    }
  };

  const fallbackCopyTextToClipboard = (textToCopy) => {
    try {
        const textarea = document.createElement("textarea");
        textarea.value = textToCopy;
        textarea.style.position = "fixed"; 
        textarea.style.opacity = "0"; 
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        setCopiedStatus(true);
        setTimeout(() => setCopiedStatus(false), 2000);
    } catch (err) {
        setError("Failed to copy text. Your browser may block clipboard access.");
    }
  };
  // --- End Copy Function ---


  // Navigation Helper
  const navigateToPage = (page) => { // Renamed from navigateTo
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleNavToSection = (ref) => {
    if (currentPage !== 'home') {
        setCurrentPage('home');
        setTimeout(() => {
            if(ref.current) ref.current.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    } else {
        if(ref.current) ref.current.scrollIntoView({ behavior: 'smooth' });
    }
  };
  
  // NEW: Animated navigation for About page
  const navigateToAbout = () => {
      const cardElement = aboutButtonRef.current;
      if (!cardElement) return navigateToPage('about'); // Fallback

      const rect = cardElement.getBoundingClientRect();
      const originalScrollTop = window.scrollY;

      // --- Phase 1: Selection Focus (200ms) ---
      // We don't focus the button, but trigger the animation directly.
      
      // Set up initial state for the ExpandingStepDetail component
      setTransitionDetails({
          id: ABOUT_ID, // Use the special ID
          rect: {
              left: rect.left,
              top: rect.top,
              width: rect.width,
              height: rect.height,
          },
          originalScrollTop: originalScrollTop
      });
      
      // --- Phase 2 & 3: Expansion and Context Shift (500ms) ---
      setTimeout(() => {
          setTransitionActive(true); 
      }, 50); // Small initial delay

      // --- Phase 4: Finalize Navigation ---
      setTimeout(() => {
          setSelectedStepId(ABOUT_ID);
          setCurrentPage('details');
          setTransitionActive(true); 
      }, 550); // 50ms + 500ms total transition time
  };


  // CRITICAL NEW LOGIC: Smooth transition function for Step Cards
  const navigateToStepDetails = (stepId) => {
    const cardElement = cardRefs.current[stepId];
    if (!cardElement) return;

    const rect = cardElement.getBoundingClientRect();
    const originalScrollTop = window.scrollY;

    // --- Phase 1: Selection Focus (200ms) ---
    setFocusedCardId(stepId);

    // Set up initial state for the ExpandingStepDetail component
    setTransitionDetails({
        id: stepId,
        rect: {
            left: rect.left,
            top: rect.top,
            width: rect.width,
            height: rect.height,
        },
        originalScrollTop: originalScrollTop
    });
    
    // --- Phase 2 & 3: Expansion and Context Shift (500ms) ---
    setTimeout(() => {
        setTransitionActive(true); 
    }, 200); // Wait for Phase 1 to complete

    // --- Phase 4: Finalize Navigation (Content Reveal starts here) ---
    setTimeout(() => {
        // Set the final state after the expansion completes
        setSelectedStepId(stepId);
        setCurrentPage('details');
        setFocusedCardId(null); // Clear focus state once navigated
        setTransitionActive(true); // Keep transitionActive true to display fully expanded content
    }, 700); // 200ms + 500ms total transition time
  };

  const handleBackTransition = () => {
    // 1. Start Shrink Animation (Phase 2 Reversal)
    setTransitionActive(false);
    
    // 2. Phase 1 Reversal: Re-focus card for visual continuity during shrink
    setFocusedCardId(transitionDetails.id); 
    
    // CRITICAL FIX: Set currentPage back to 'home' immediately to start fading in the main content
    setCurrentPage('home');

    // 3. Wait for shrink (500ms) and clean up is handled by useEffect
  };


  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setResult(null);
    setError(null);
    // Clear document selection when switching away
    if (tab !== 'document' && selectedFile) {
        setSelectedFile(null);
        if (fileInputRef.current) fileInputRef.current.value = null;
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      // Increased size limit from 5MB to 10MB
      if (file.size > 10 * 1024 * 1024) { 
        setError("File size exceeds 10MB limit. Please upload a smaller document/image.");
        setSelectedFile(null);
        e.target.value = null;
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };
  
  // --- Enhanced Error Check for Local Server Status ---
  const handleFetchError = (err) => {
    if (err instanceof TypeError && (err.message === 'Failed to fetch' || err.message.includes('ERR_CONNECTION_REFUSED'))) {
      setError("❌ Connection Error: Make sure your Python backend is running locally on http://localhost:8000!");
    } else {
      // Catch specific 403 error for better user feedback
      if (err.message.includes('403')) {
          setError("❌ LLM API Error: Permission Denied. Your backend is running, but the API key is not being read/injected correctly by the Canvas environment. See console for details.");
      } else {
          // Catch 429 Quota Exceeded error
          if (err.message.includes('429') && err.message.includes('Quota exceeded')) {
               setError("⚠️ Quota Exceeded (429): You've hit the daily free-tier limit for the Claim Analyzer's API. Please wait a few hours or until the quota resets.");
          } else {
               setError(`Analysis Error: ${err.message}. Check your local backend logs.`);
          }
      }
    }
  };
  // ----------------------------------------------------


  const analyzeText = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    setCopiedStatus(false);
    try {
      const apiUrl = `${API_BASE_URL}/analyze`;
      const response = await fetch(apiUrl, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText, debug: debugMode }),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned error status: ${response.status}`);
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      handleFetchError(err);
    } finally {
      setLoading(false);
    }
  };

  const analyzeUrl = async () => {
    if (!inputUrl.trim()) return;
    setLoading(true);
    setError(null);
    setCopiedStatus(false);
    try {
      const response = await fetch(`${API_BASE_URL}/predict_url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: inputUrl, debug: debugMode }),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned error status: ${response.status}`);
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      handleFetchError(err);
    } finally {
      setLoading(false);
    }
  };

  // Renamed from analyzeImage
  const analyzeDocument = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setCopiedStatus(false);
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        // Note: For file uploads, query params are usually handled via URL if form data is used, 
        // but since we are modifying endpoints in main.py, we might need to adjust or rely on params if supported.
        // The current backend for file upload doesn't take 'debug' via form fields easily without changing Pydantic model or params.
        // For simplicity, we assume text/url endpoints are primary for debugging now, or append to URL.
        
        let url = `${API_BASE_URL}/predict_image`;
        if (debugMode) {
             // Append debug param if your backend supports it as query param or handles it in form data (requires backend change)
             // Based on provided main.py, predict_image takes File, not a model with debug.
             // So debug might not be supported for image unless added to backend signature.
             // We will leave it for now or append query string if backend was updated to accept it.
             // url += "?debug=true"; 
        }
        
        const response = await fetch(url, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server returned error status: ${response.status}`);
        }
        
        const data = await response.json();
        setResult(data);
        
    } catch (err) {
      handleFetchError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (activeTab === 'text') analyzeText();
    if (activeTab === 'url') analyzeUrl();
    // Use the new document analysis function
    if (activeTab === 'document') analyzeDocument();
  };

  // Helper to determine styles based on classification
  const getResultStyles = (classification) => {
    // We now use result.confidence_label from the backend to display the specific description
    const label = result?.confidence_label || "Unverified / Needs Sources";

    if (classification === 'Fake') {
        return {
            container: 'bg-rose-50 border-rose-100 text-rose-700',
            shadow: 'shadow-rose-500/10',
            bar: 'bg-rose-600',
            icon: <AlertTriangle className="w-6 h-6" />,
            label: label
        };
    } else if (classification === 'Real') {
        return {
            container: 'bg-blue-50 border-blue-100 text-blue-700',
            shadow: 'shadow-blue-500/10',
            bar: 'bg-blue-600',
            icon: <ShieldCheck className="w-6 h-6" />,
            label: label
        };
    } else if (classification === 'Biased') {
         return {
            container: 'bg-orange-50 border-orange-100 text-orange-700',
            shadow: 'shadow-orange-500/10',
            bar: 'bg-orange-500',
            icon: <AlertCircle className="w-6 h-6" />,
            label: label
        };
    } else { // Unverified / Unsure
        return {
            container: 'bg-amber-50 border-amber-100 text-amber-700',
            shadow: 'shadow-amber-500/10',
            bar: 'bg-amber-500',
            icon: <HelpCircle className="w-6 h-6" />,
            label: label
        };
    }
  };
  
  // --- NEW COMPONENT: SIGNAL CHIP (COLOR CODED) ---
  const SignalPill = ({ label, value }) => {
    // 1. Core Color System (Accessible Shades)
    const COLORS = {
        // Success / Safe (Soft Teal/Cyan Green)
        LOW: { bg: '#E6FFFA', text: '#0D9488', border: '#34D399' }, 
        // Caution / Warning (Soft Gold/Amber)
        MEDIUM: { bg: '#FFF7E6', text: '#D97706', border: '#FCD34D' }, 
        // Danger / High Risk (Soft Rose/Red)
        HIGH: { bg: '#FFF0F5', text: '#BE185D', border: '#F472B6' }, 
        // Neutral / Not Present (Light Gray/Slate)
        NONE: { bg: '#F3F4F6', text: '#4B5563', border: '#9CA3AF' }, 
    };

    // 2. Determine Color Mapping based on Polarity
    let colorSet = COLORS.NONE;
    const val = value.toUpperCase();

    // Positive Polarity: HIGH = Good (LOW/Green), LOW = Bad (HIGH/Red)
    const isPositivePolarity = label === 'plausibility' || label === 'evidence_specificity';
    
    // Logic for Positive Polarity (Plausibility, Evidence)
    if (isPositivePolarity) {
        if (val === 'HIGH' || val === 'SPECIFIC') colorSet = COLORS.LOW; // High Plausibility = Good
        else if (val === 'MEDIUM' || val === 'VAGUE') colorSet = COLORS.MEDIUM; // Mid/Vague = Warning
        else if (val === 'LOW' || val === 'NONE') colorSet = COLORS.HIGH; // Low/None = Danger
    } 
    // Logic for Negative Polarity (Bias, Uncertainty)
    else {
        if (val === 'LOW' || val === 'NONE') colorSet = COLORS.LOW; // Low Bias/No Uncertainty = Good
        else if (val === 'MEDIUM') colorSet = COLORS.MEDIUM; // Medium Bias = Warning
        else if (val === 'HIGH' || val === 'YES') colorSet = COLORS.HIGH; // High Bias/Yes Uncertainty = Danger
    }
    
    // 3. Define Tailwind classes using arbitrary values
    const styleClass = `bg-[${colorSet.bg}] text-[${colorSet.text}] border-[${colorSet.border}]`;

    // Optional Accent Colors for Label Prefix
    const accentColors = {
        'plausibility': 'text-indigo-600', 
        'evidence_specificity': 'text-cyan-600', 
        'bias_level': 'text-purple-600', 
        'uncertainty_present': 'text-gray-600', 
    };

    const accentColorClass = accentColors[label] || 'text-gray-600';
    
    // Format label for display
    const displayLabel = label.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');

    return (
        <div className={`text-xs font-medium px-3 py-1 rounded-full border ${styleClass} flex items-center gap-1.5 shadow-sm`}>
            <span className={`${accentColorClass} font-bold opacity-90`}>{displayLabel}:</span>
            <span className="uppercase tracking-wide">{val}</span>
        </div>
    );
  };
  // ------------------------------------------

  const displayScore = result ? result.credibility_score : 0;
  const resultStyle = result ? getResultStyles(result.classification) : null;
  const claimsData = result?.claims_data?.claims || [];
  
  // NEW: Document Metadata Extraction
  const documentType = result?.document_type;
  const extractionMethod = result?.text_extraction_method;


  return (
    <div 
      className="min-h-screen bg-gradient-to-br from-gray-50 to-indigo-50 text-gray-900 font-sans relative overflow-x-hidden selection:bg-cyan-100 selection:text-indigo-900"
      style={{ fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' }}
    >
      
      {/* Background Ambience */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
         <div className="absolute top-[-10%] left-[-10%] w-[50vw] h-[50vw] bg-indigo-200/20 rounded-full blur-[120px]" />
         <div className="absolute bottom-[-10%] right-[-10%] w-[50vw] h-[50vw] bg-cyan-100/40 rounded-full blur-[120px]" />
      </div>

      {/* --- STEP TRANSITION OVERLAY (Fixed Position) --- */}
      {/* This renders only the expanding content when transitionDetails is available */}
      {transitionDetails && (
        <ExpandingStepDetail 
            transitionDetails={transitionDetails} 
            isActive={transitionActive}
            handleBackTransition={handleBackTransition}
        />
      )}
      {/* --- END TRANSITION OVERLAY --- */}


      {/* Navbar */}
      <nav className="fixed top-0 w-full z-50 bg-white border-b border-gray-200 shadow-sm transition-all duration-300">
        <div className="max-w-5xl mx-auto px-6 h-20 flex items-center justify-between">
          
          <div className="flex items-center gap-3 group">
            <button className="flex items-center gap-3 cursor-pointer" onClick={() => navigateToPage('home')} title="Go to Analyzer">
                <div className="w-10 h-10 bg-indigo-700 rounded-lg flex items-center justify-center shadow-md shadow-indigo-200 group-hover:bg-indigo-800 transition-colors">
                  <ShieldCheck className="w-6 h-6 text-white" />
                </div>
                <span className="text-xl font-bold text-gray-900 tracking-tight group-hover:text-indigo-700 transition-colors">
                  Claim <span className="text-cyan-500">Analyzer</span>
                </span>
            </button>
            
            {/* NEW: ABOUT BUTTON */}
            <button 
                ref={aboutButtonRef} // Added Ref for capturing position
                onClick={navigateToAbout} // Changed to animated navigation handler
                className={`text-sm font-semibold transition-colors py-1 px-3 rounded-full hover:bg-gray-100
                  ${currentPage === 'about' ? 'text-indigo-700 bg-gray-100' : 'text-gray-600 hover:text-indigo-700'}
                `}
                title="View project details"
            >
                About
            </button>
          </div>

          <div className="hidden md:flex items-center gap-8">
            <button onClick={() => navigateToPage('home')} className={`text-sm font-semibold transition-colors ${currentPage === 'home' ? 'text-indigo-700' : 'text-gray-600 hover:text-indigo-700'}`}>Analyze</button>
            <button onClick={() => handleNavToSection(howItWorksRef)} className="text-sm font-semibold text-gray-600 hover:text-indigo-700 transition-colors">How It Works</button>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-gray-800 transition-colors" title="View on GitHub">
              <Github className="w-5 h-5" />
            </a>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      {/* Conditional rendering based on currentPage to hide content during detailed view transition */}
      <main className={`max-w-3xl mx-auto px-6 pt-20 pb-20 transition-opacity duration-300 ${currentPage === 'details' ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
        
        {currentPage === 'home' && (
            <>
                {/* Minimalist Hero */}
                <div className="text-center mb-16 pt-16 animate-in fade-in slide-in-from-bottom-2 duration-700 ease-out fill-mode-forwards">
                  <h1 className="text-5xl md:text-6xl font-black text-gray-900 mb-6 tracking-tight leading-tight">
                    Decompose Claims. <br className="hidden md:block"/> Get Signals.
                  </h1>
                  <p className="text-gray-600 text-xl font-medium max-w-2xl mx-auto leading-relaxed mb-4">
                    Uses the Claim Analyzer to break down text into atomic assertions and annotate their credibility signals.
                  </p>
                </div>

                {/* Input Section */}
                <div className="bg-white/60 backdrop-blur-xl rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-white/60 ring-1 ring-gray-100/50 relative overflow-hidden transition-all duration-300 animate-in fade-in slide-in-from-bottom-3 duration-1000 delay-150 fill-mode-forwards">
                    
                    <div className="text-center pt-8 pb-0">
                      <h2 className="text-2xl font-bold text-gray-900 tracking-tight">Analyze Information</h2>
                    </div>

                    {/* Updated Tabs: 'image' -> 'document' */}
                    <div className="mx-8 mt-6 relative flex border-b border-gray-200/70">
                      {['text', 'url', 'document'].map((tab) => (
                        <button
                          key={tab}
                          onClick={() => handleTabChange(tab)}
                          className={`flex-1 py-4 flex items-center justify-center gap-2 text-sm font-semibold transition-colors duration-[160ms] ease-out relative z-10 group
                            ${activeTab === tab 
                              ? 'text-indigo-700' 
                              : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50/30 rounded-t-lg'}
                          `}
                        >
                          <span className={`transition-opacity duration-200 ${activeTab === tab ? 'opacity-100' : 'opacity-60 group-hover:opacity-100'}`}>
                            {tab === 'text' && <Type className="w-4 h-4" />}
                            {tab === 'url' && <Link className="w-4 h-4" />}
                            {/* Updated icon for document */}
                            {tab === 'document' && <FileText className="w-4 h-4" />}
                          </span>
                          <span className="capitalize">{tab}</span>
                        </button>
                      ))}
                      
                      <div 
                        className="absolute bottom-0 h-[2px] bg-indigo-600 transition-all duration-[180ms] ease-out z-20"
                        style={{
                            width: `${100 / 3}%`,
                            left: activeTab === 'text' ? '0%' : activeTab === 'url' ? `${100 / 3}%` : `${200 / 3}%`
                        }}
                      />
                    </div>

                    <div className="p-8 pt-6">
                      <form onSubmit={handleSubmit} className="space-y-6">
                        <div key={activeTab} className="animate-in fade-in duration-150 ease-out">
                            {activeTab === 'text' && (
                                <div className="relative">
                                    <textarea
                                    className="w-full h-40 p-4 rounded-xl bg-gray-50/50 border border-gray-200 focus:bg-white focus:border-indigo-500/30 focus:ring-4 focus:ring-indigo-500/10 outline-none resize-none text-gray-700 placeholder:text-gray-400 transition-all text-base leading-relaxed shadow-inner pr-12"
                                    placeholder="Paste the news article or statement here"
                                    value={inputText}
                                    onChange={(e) => setInputText(e.target.value)}
                                    />
                                    {inputText && (
                                        <button
                                            type="button"
                                            onClick={() => setInputText('')}
                                            className="absolute top-4 right-4 text-gray-400 hover:text-gray-700 transition-colors focus:outline-none"
                                            title="Clear text"
                                        >
                                            <X className="w-5 h-5" />
                                        </button>
                                    )}
                                </div>
                            )}

                            {activeTab === 'url' && (
                                <div className="relative">
                                    <input
                                    type="url"
                                    className="w-full p-4 rounded-xl bg-gray-50/50 border border-gray-200 focus:bg-white focus:border-indigo-500/30 focus:ring-4 focus:ring-indigo-500/10 outline-none text-gray-700 placeholder:text-gray-400 transition-all text-base shadow-inner pr-12"
                                    placeholder="Enter the link to the news source"
                                    value={inputUrl}
                                    onChange={(e) => setInputUrl(e.target.value)}
                                    />
                                    {inputUrl && (
                                        <button
                                            type="button"
                                            onClick={() => setInputUrl('')}
                                            className="absolute top-1/2 right-4 transform -translate-y-1/2 text-gray-400 hover:text-gray-700 transition-colors focus:outline-none"
                                            title="Clear URL"
                                        >
                                            <X className="w-5 h-5" />
                                        </button>
                                    )}
                                </div>
                            )}
                            
                            {/* Updated Tab: 'image' -> 'document' */}
                            {activeTab === 'document' && (
                                <div className="space-y-4">
                                    <div className="relative">
                                        <input
                                            type="file"
                                            // Allow image file types for document analysis
                                            accept="image/*,.pdf"
                                            ref={fileInputRef}
                                            onChange={handleFileChange}
                                            className="hidden"
                                            id="document-upload"
                                        />
                                        <label 
                                            htmlFor="document-upload" 
                                            className={`flex flex-col items-center justify-center w-full h-40 p-4 rounded-xl border-2 border-dashed transition-colors cursor-pointer 
                                                ${selectedFile 
                                                    ? 'bg-emerald-50 border-emerald-300 text-emerald-800' 
                                                    : 'bg-gray-50/50 border-gray-300 hover:border-indigo-500/50 text-gray-500 hover:text-indigo-600'}`
                                            }
                                        >
                                            <FileText className="w-6 h-6 mb-2" />
                                            {selectedFile ? (
                                                <span className="font-semibold text-sm">{selectedFile.name} (Ready for OCR analysis)</span>
                                            ) : (
                                                <span className="font-medium text-sm">Click to upload Document (Image/PDF) (Max 10MB)</span>
                                            )}
                                            {selectedFile && (
                                                <button
                                                    type="button"
                                                    onClick={(e) => { e.preventDefault(); setSelectedFile(null); if (fileInputRef.current) fileInputRef.current.value = null; }}
                                                    className="text-xs text-rose-500 hover:text-rose-700 mt-1"
                                                >
                                                    Clear Document
                                                </button>
                                            )}
                                        </label>
                                    </div>
                                    
                                    {/* Removed manual crop input for simplicity */}
                                    {/* <div className="relative text-sm">...</div> */}
                                </div>
                            )}
                        </div>

                        {/* Debug Toggle */}
                        <div className="flex items-center gap-2 mb-2">
                             <input 
                               type="checkbox" 
                               id="debugToggle" 
                               checked={debugMode}
                               onChange={(e) => setDebugMode(e.target.checked)}
                               className="w-4 h-4 text-indigo-600 rounded focus:ring-indigo-500 border-gray-300"
                             />
                             <label htmlFor="debugToggle" className="text-sm text-gray-600 font-medium cursor-pointer select-none flex items-center gap-1">
                                <Bug className="w-3 h-3 text-gray-400"/> Enable Debug Info (Backend)
                             </label>
                        </div>

                        <button
                          type="submit"
                          disabled={loading || (activeTab === 'text' && !inputText) || (activeTab === 'url' && !inputUrl) || (activeTab === 'document' && !selectedFile)}
                          className="w-full bg-indigo-700 hover:bg-indigo-600 disabled:bg-gray-200 disabled:text-gray-400 text-white font-bold py-4 rounded-xl transition-all duration-200 flex flex-col items-center justify-center gap-1 shadow-md shadow-indigo-900/10 hover:shadow-lg hover:shadow-indigo-900/20 hover:-translate-y-px active:translate-y-px focus:ring-4 focus:ring-indigo-500/20 outline-none min-h-[64px]"
                        >
                          {loading ? (
                            <>
                              <div className="flex items-center gap-2">
                                 <Loader2 className="animate-spin w-5 h-5" />
                                 <span className="font-semibold text-base">
                                     {activeTab === 'document' 
                                         ? 'Running Local OCR & NN Analysis...' 
                                         : 'Processing Claim Signals via Hybrid Engine...'}
                                 </span>
                              </div>
                              <span className="text-xs font-normal opacity-80">Extracting claims and calculating credibility vectors</span>
                            </>
                          ) : (
                            "Analyze Claims & Get Signals"
                          )}
                        </button>
                      </form>

                      {error && (
                        <div className="mt-6 p-4 bg-rose-50 border border-rose-100 text-rose-600 rounded-xl flex items-center gap-3 text-sm animate-in fade-in slide-in-from-top-2">
                          <AlertCircle className="w-5 h-5 shrink-0" />
                          {error}
                        </div>
                      )}
                    </div>
                </div>

                {/* Results Area */}
                {result && resultStyle && (
                  <div className="mt-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
                    <div className={`p-1.5 rounded-3xl bg-white/55 backdrop-blur-md border border-white/60 shadow-xl ${resultStyle.shadow}`}>
                        <div className="bg-white rounded-[22px] p-8 md:p-10 border border-gray-100">
                            
                            <div className="text-center mb-8">
                               <h3 className="text-2xl font-bold text-gray-900 mb-2">Claim Signal Breakdown</h3>
                               <p className="text-gray-500 text-sm">Detailed analysis provided by Claim Analyzer.</p>
                            </div>
                            
                            {/* NEW: SCIENTIFIC CONTRADICTION BANNER (FKC) */}
                            {/* FIX 3: FRONTEND SAFETY GUARD - Only show banner if claim is valid (not placeholder) */}
                            {result.fundamental_contradiction && 
                             !result.claims_data?.claims?.some(c => c.claim_text.includes("No substantial claim found")) && (
                                <div className="mb-8 p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 flex items-start gap-3 shadow-sm">
                                    <AlertTriangle className="w-6 h-6 shrink-0 mt-0.5" />
                                    <div>
                                        <h4 className="font-bold text-lg">Scientific Contradiction Detected</h4>
                                        <p className="text-sm opacity-90">
                                            The Fundamental Knowledge Checker (FKC) detected a conflict with established scientific axioms. 
                                            This is a major warning sign, though the HCDM score remains independent.
                                        </p>
                                    </div>
                                </div>
                            )}

                            {/* Document Analysis Metadata Display (Only visible for document type) */}
                            {documentType && (
                                <div className="mb-8 p-4 bg-gray-50 rounded-xl border border-gray-200 text-sm">
                                    <div className="flex items-center gap-2 font-semibold text-gray-700 mb-2">
                                        <BookOpen className="w-4 h-4 text-cyan-600" />
                                        Document Analysis Report
                                    </div>
                                    <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                                        <div>
                                            <span className="font-medium text-gray-900">Type Detected:</span> {documentType.replace('_', ' ').toUpperCase()}
                                        </div>
                                        <div>
                                            <span className="font-medium text-gray-900">Extraction Method:</span> {extractionMethod?.toUpperCase()}
                                        </div>
                                    </div>
                                    {extractionMethod === 'failed' && (
                                        <p className="mt-2 text-rose-500 italic">Extraction Failed: Check dependencies (PyMuPDF for PDF, or VLM model state).</p>
                                    )}
                                </div>
                            )}

                            {/* OVERALL CREDIBILITY SCORE */}
                            <div className="flex flex-col items-center justify-center gap-4 mb-10 p-4 bg-gray-50 rounded-xl border border-gray-200">
                                 {/* Display Confidence Label */}
                                 <div className={`flex items-center gap-3 px-6 py-3 rounded-full border-2 ${resultStyle.container}`}>
                                     {resultStyle.icon}
                                     <span className="text-lg font-bold tracking-tight">
                                        Overall Judgment: {result.confidence_label || resultStyle.label} 
                                     </span>
                                 </div>
                                 <div className="text-gray-500 font-medium">
                                    Heuristic Credibility Score: <span className="text-gray-900 font-bold">{displayScore}/100</span>
                                 </div>
                            </div>
                            
                            {/* CLAIMS ITERATION SECTION - IMPROVED STYLING */}
                            {claimsData.length > 0 ? (
                                <div className="space-y-6">
                                    <h4 className="text-xl font-bold text-indigo-700 flex items-center gap-2 mb-4">
                                        <Terminal className="w-5 h-5"/>
                                        Decomposed Claims ({claimsData.length})
                                    </h4>
                                    
                                    {claimsData.map((claim, index) => (
                                        // Highlight claims with FKC warning - FIX 3: Safety Guard on individual card
                                        <div key={index} className={`p-5 bg-white border rounded-xl shadow-sm ${claim.fkc_flag && !claim.claim_text.includes("No substantial claim found") ? 'border-rose-200 ring-1 ring-rose-100' : 'border-gray-200'}`}>
                                            <div className="mb-3 flex items-center justify-between">
                                                <span className="font-mono text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded-full">CLAIM {index + 1}</span>
                                                {claim.fkc_flag && !claim.claim_text.includes("No substantial claim found") && (
                                                    <span className="text-xs font-bold text-rose-600 flex items-center gap-1">
                                                        <AlertTriangle className="w-3 h-3"/> CONTRADICTION
                                                    </span>
                                                )}
                                            </div>
                                            <p className="text-base text-gray-800 leading-relaxed font-medium">
                                                 {claim.claim_text}
                                            </p>
                                            
                                            {/* FKC Reason display inside claim card - FIX 3: Safety Guard */}
                                            {claim.fkc_reason && !claim.claim_text.includes("No substantial claim found") && (
                                                <div className="mt-3 text-xs text-rose-700 bg-rose-50 p-2 rounded border border-rose-100">
                                                    <strong>FKC Warning:</strong> {claim.fkc_reason}
                                                </div>
                                            )}

                                            <div className="flex flex-wrap gap-2 pt-4 border-t border-gray-100 mt-4">
                                                {Object.entries(claim.signals).map(([key, value]) => (
                                                    <SignalPill key={key} label={key} value={value} />
                                                ))}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="p-6 bg-amber-50 rounded-xl text-amber-800 text-center">
                                    No clear atomic claims could be extracted by the Claim Analyzer for analysis.
                                    {/* Additional context for document analysis failure */}
                                    {activeTab === 'document' && <p className="text-sm mt-2">Try re-uploading with a clearer document or less text.</p>}
                                </div>
                            )}

                            {/* FACTORS (LLM Signal Score Drivers) */}
                            <div className="bg-gray-50 rounded-2xl p-6 border border-gray-200 mt-10">
                                <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
                                    <Brain className="w-4 h-4 text-indigo-700"/>
                                    Score Driver Factors
                                </h4>
                                <ul className="space-y-3 text-sm text-gray-600">
                                    {result.factors && result.factors.length > 0 ? (
                                        result.factors.map((factor, idx) => (
                                            <li key={idx} className="flex items-start gap-2">
                                                <span className="text-lg leading-none mt-0.5" role="img" aria-label="indicator">{factor.startsWith('🚩') ? '🚩' : factor.startsWith('✅') ? '✅' : 'ℹ️'}</span>
                                                <span className={factor.startsWith('🚩') ? 'text-rose-700' : factor.startsWith('✅') ? 'text-emerald-700' : 'text-gray-600'}>
                                                    {factor.replace(/^[🚩✅ℹ️]\s*/, '')}
                                                </span>
                                            </li>
                                        ))
                                    ) : (
                                        <li className="flex items-start gap-2 text-gray-400">
                                            No explicit factors were generated.
                                        </li>
                                    )}
                                </ul>
                            </div>
                            
                            {/* DEBUG BOX RENDER (Only visible if debugMode was enabled AND data exists) */}
                            {result.debug_box && Object.keys(result.debug_box).length > 0 && (
                                <div className="bg-slate-900 text-slate-200 rounded-2xl p-6 mt-10 border border-slate-700 overflow-hidden shadow-lg">
                                    <h4 className="font-bold text-white mb-4 flex items-center gap-2 border-b border-slate-700 pb-3">
                                        <Bug className="w-4 h-4 text-cyan-400"/>
                                        Backend Debug Signals (Pre-HCDM)
                                    </h4>
                                    
                                    <div className="space-y-6 text-xs font-mono">
                                        {/* CRCS Data */}
                                        {result.debug_box.crcs_consensus && (
                                            <div>
                                                <span className="text-cyan-400 font-bold block mb-1">CRCS Consensus (Mock):</span>
                                                <div className="bg-slate-800 p-3 rounded border border-slate-700">
                                                    <div>ACS: <span className="text-white">{result.debug_box.crcs_consensus.acs}</span></div>
                                                    <div>Label: <span className="text-white">{result.debug_box.crcs_consensus.consensus_label}</span></div>
                                                    <div>Evidence Count: <span className="text-white">{result.debug_box.crcs_consensus.evidence_count}</span></div>
                                                </div>
                                            </div>
                                        )}

                                        {/* FKC Checks */}
                                        {result.debug_box.fkc_checks && result.debug_box.fkc_checks.length > 0 && (
                                            <div>
                                                <span className="text-rose-400 font-bold block mb-1">FKC Invariant Checks:</span>
                                                <div className="space-y-2">
                                                    {result.debug_box.fkc_checks.map((check, idx) => (
                                                        <div key={idx} className="bg-slate-800 p-3 rounded border border-slate-700">
                                                            <div className="opacity-70 mb-1">"{check.claim.substring(0, 60)}..."</div>
                                                            <div className="flex gap-4">
                                                                <span>Violation: <span className={check.violation ? "text-rose-400" : "text-emerald-400"}>{String(check.violation)}</span></span>
                                                                {check.warning && <span className="text-amber-400">"{check.warning}"</span>}
                                                            </div>
                                                            {check.debug_info && Object.keys(check.debug_info).length > 0 && (
                                                                <div className="mt-1 opacity-60">{JSON.stringify(check.debug_info)}</div>
                                                            )}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        {/* CARS Retrieval */}
                                        {result.debug_box.cars_retrieval && (
                                            <div>
                                                <span className="text-purple-400 font-bold block mb-1">CARS Retrieval (Contexts):</span>
                                                <div className="bg-slate-800 p-3 rounded border border-slate-700">
                                                    <div>Retrieved: <span className={result.debug_box.cars_retrieval.retrieved ? "text-emerald-400" : "text-slate-400"}>{String(result.debug_box.cars_retrieval.retrieved)}</span></div>
                                                    {result.debug_box.cars_retrieval.sources && (
                                                        <div className="mt-1">Sources: {result.debug_box.cars_retrieval.sources.join(", ")}</div>
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                            
                            {/* Analysis Summary - IMPROVED STYLING (Glassmorphic) */}
                            {result.explanation && (
                                <div className="bg-indigo-100/70 backdrop-blur-sm text-indigo-900 rounded-2xl p-6 border border-indigo-200/50 mt-6 relative shadow-lg shadow-indigo-900/10">
                                    <h4 className="font-bold text-indigo-800 mb-2 text-center text-xl tracking-wide flex items-center justify-center gap-2">
                                        <ShieldCheck className="w-5 h-5"/>
                                        Analysis Summary
                                        <button
                                            onClick={() => copyToClipboard(result.explanation)}
                                            className="absolute top-4 right-4 p-1 rounded-lg text-indigo-600 hover:bg-indigo-200 transition-colors focus:outline-none focus:ring-2 focus:ring-white/50"
                                            title="Copy Summary"
                                        >
                                            {copiedStatus ? (
                                                <CheckCircle className="w-5 h-5 text-emerald-600" />
                                            ) : (
                                                <Copy className="w-5 h-5" />
                                            )}
                                        </button>
                                    </h4>
                                    <blockquote className="text-lg leading-relaxed font-medium mt-4 italic border-t border-indigo-300/50 pt-4">
                                        "{result.explanation}"
                                    </blockquote >
                                </div>
                            )}

                            {/* Improvement Suggestion - IMPROVED STYLING */}
                            {result.suggestion && (
                                <div className="bg-amber-50 rounded-2xl p-6 border border-amber-200 mt-6 text-left shadow-sm">
                                    <h4 className="font-bold text-amber-900 mb-3 flex items-center gap-2">
                                        <Lightbulb className="w-5 h-5 text-amber-600" />
                                        Improvement Suggestion (for the author)
                                    </h4>
                                    <p className="text-amber-800 text-sm leading-relaxed">
                                        {result.suggestion}
                                    </p>
                                </div>
                            )}
                            
                            {/* Related News and Verification Tools Sections */}
                            <div className="grid md:grid-cols-2 gap-6 mt-6">
                                
                                {/* Related News Section (New) */}
                                {result.related_news && result.related_news.length > 0 && (
                                    <div className="p-4 bg-white border border-gray-200 rounded-xl shadow-sm">
                                        <h5 className="font-bold text-gray-800 mb-2 flex items-center gap-2 text-sm">
                                            <Globe className="w-4 h-4 text-cyan-500" />
                                            Related Contextual News
                                        </h5>
                                        <ul className="space-y-1 text-xs text-gray-600">
                                            {result.related_news.map((item, index) => (
                                                <li key={index}>
                                                    <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">
                                                        {item.title.length > 50 ? item.title.substring(0, 50) + '...' : item.title}
                                                    </a>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* Verification Tools Section (From Backend Placeholder) */}
                                {result.verification_tools && result.verification_tools.length > 0 && (
                                    <div className="p-4 bg-white border border-gray-200 rounded-xl shadow-sm">
                                        <h5 className="font-bold text-gray-800 mb-2 flex items-center gap-2 text-sm">
                                            <CheckCircle className="w-4 h-4 text-emerald-500" />
                                            External Verification Tools
                                        </h5>
                                        <ul className="space-y-1 text-xs text-gray-600">
                                            {result.verification_tools.map((tool, index) => (
                                                <li key={index}>
                                                    <a href={tool.url} target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">
                                                        {tool.source}
                                                        <ExternalLink className="w-3 h-3 inline ml-1 align-baseline"/>
                                                    </a>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                            </div>
                            
                            <div className="text-center mt-6">
                                <p className="text-xs text-gray-400 max-w-sm mx-auto">
                                    The overall credibility score is derived from aggregating the LLM's signals using a weighted heuristic model.
                                </p>
                            </div>
                        </div>
                    </div>
                  </div>
                )}
                
                {/* How It Works Section */}
                <div className="mt-24 mb-20 scroll-mt-24" ref={howItWorksRef}>
                    <div className="text-center mb-12">
                        <h2 className="2xl font-bold text-gray-900">How the System Works</h2>
                        <p className="text-gray-500">The LLM-Hybrid Claim Analysis Pipeline</p>
                    </div>
                    
                    <div className="grid md:grid-cols-4 gap-6 relative">
                        <div className="hidden md:block absolute top-8 left-0 w-full h-0.5 bg-gray-100 -z-10 transform scale-x-90"></div>
                        
                        {Object.entries(STEP_CONTENT).map(([id, step], idx) => {
                            const IconComponent = step.icon;
                            const isFocused = focusedCardId === id;
                            const isDimmed = focusedCardId && focusedCardId !== id;

                            return (
                                // CRITICAL CHANGE: Added ref and managed focus/dimming for Phase 1
                                <button 
                                    key={id} 
                                    ref={el => cardRefs.current[id] = el}
                                    onClick={() => navigateToStepDetails(id)}
                                    className={`
                                        bg-white p-6 rounded-2xl border border-gray-100 text-center shadow-[0_4px_20px_rgb(0,0,0,0.03)] transition-all duration-200 transform-gpu cursor-pointer group 
                                        ${isFocused ? 'scale-[1.03] shadow-lg ring-2 ring-indigo-500/50' : ''}
                                        ${isDimmed ? 'opacity-30 blur-sm pointer-events-none' : 'hover:-translate-y-1 hover:shadow-[0_8px_30px_rgb(0,0,0,0.06)]'}
                                    `}
                                >
                                    <div className="w-16 h-16 mx-auto bg-gray-50 rounded-full flex items-center justify-center mb-4 text-cyan-500 border border-gray-100 transition-all group-hover:bg-indigo-100 group-hover:text-indigo-700">
                                        <IconComponent className="w-8 h-8" />
                                    </div>
                                    <h3 className="font-bold text-gray-900 mb-2">{id}</h3>
                                    <p className="text-sm text-gray-500">{step.detail.split('\n')[0].substring(0, 50)}...</p>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* NEW: TECHNOLOGIES SECTION */}
                <div className="mt-24 mb-20 scroll-mt-24">
                    <div className="text-center mb-12">
                        <h2 className="text-2xl font-bold text-gray-900">Core Technologies</h2>
                        <p className="text-gray-500">The essential tools powering the Claim Analyzer engine.</p>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                        {TECHNOLOGIES.map((tech) => {
                            const IconComponent = tech.icon;
                            return (
                                <a 
                                    key={tech.name} 
                                    href={tech.link} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="block p-4 bg-white rounded-xl border border-gray-100 text-center shadow-sm hover:shadow-md hover:border-indigo-200 transition-all hover:-translate-y-0.5"
                                >
                                    <div className="w-12 h-12 mx-auto bg-indigo-50 rounded-full flex items-center justify-center text-indigo-600 border border-indigo-100">
                                        <IconComponent className="w-6 h-6" />
                                    </div>
                                    <p className="text-sm font-semibold text-gray-800">{tech.name}</p>
                                    <p className="text-xs text-indigo-500 hover:underline">View Site</p>
                                </a>
                            );
                        })}
                    </div>
                </div>

            </>
        )}
        
        {/* NEW: ABOUT PAGE RENDER */}
        {currentPage === 'about' && <AboutPage closePage={navigateToPage('home')} />}
        
        {/* The details page content is rendered by the ExpandingStepDetail component when transitionDetails is set */}

      </main>

      {/* Footer (omitted for brevity) */}
      <footer className="bg-white border-t border-gray-200 py-12">
          <div className="max-w-5xl mx-auto px-6 text-center">
              <div className="flex items-center justify-center gap-2 mb-4 opacity-50">
                  <ShieldCheck className="w-5 h-5 text-gray-600" />
                  <span className="font-bold text-gray-600">Claim Analyzer</span>
              </div>
              <p className="text-gray-400 text-sm mb-6">
                  © 2025 Claim Analyzer. Powered by Claim Analyzer and custom heuristic logic.
              </p>
          </div>
      </footer>

    </div>
  );
}