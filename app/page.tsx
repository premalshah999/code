"use client"

import Link from "next/link"
import { useEffect, useRef, useState } from "react"
import ShaderBackground from "@/components/shader-background"
import { Terminal } from "lucide-react"
import {
  SiPython,
  SiJavascript,
  SiTypescript,
  SiMysql,
  SiHtml5,
  SiCss3,
  SiFastapi,
  SiReact,
  SiNodedotjs,
  SiPostgresql,
  SiMongodb,
  SiSpringboot,
  SiFlask,
  SiDjango,
  SiExpress,
  SiNextdotjs,
  SiVite,
  SiAmazon,
  SiGooglecloud,
  SiDocker,
  SiKubernetes,
  SiGithubactions,
  SiJenkins,
  SiMlflow,
  SiTensorflow,
  SiPytorch,
  SiScikitlearn,
  SiPandas,
  SiNumpy,
  SiOpencv,
  SiHuggingface,
  SiLangchain,
} from "react-icons/si"

export default function Home() {
  const [activeSection, setActiveSection] = useState("")
  const [activeAboutTab, setActiveAboutTab] = useState("outside-work")
  const [activeSkillsTab, setActiveSkillsTab] = useState("Languages")
  const [selectedSkill, setSelectedSkill] = useState<{
    name: string
    proficiency: number
    description: string
    usedIn: string
  } | null>(null)
  const [expandedSkills, setExpandedSkills] = useState<{
    [key: string]: boolean
  }>({
    Languages: false,
    "Development & Databases": false,
    "Cloud & DevOps": false,
    "Machine Learning & AI": false,
  })
  const [displayedName, setDisplayedName] = useState("")
  const [nameTypingDone, setNameTypingDone] = useState(false)
  const fullName = "Premal Shah"
  const roles = ["Data Science", "Full Stack Development", "Machine Learning"]
  const [currentRoleIndex, setCurrentRoleIndex] = useState(0)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [selectedExperience, setSelectedExperience] = useState<any | null>(null)
  const [selectedProject, setSelectedProject] = useState<any | null>(null)
  const [selectedResearch, setSelectedResearch] = useState<any | null>(null)

  // Typing effect - runs once on mount
  useEffect(() => {
    let index = 0
    const interval = setInterval(() => {
      if (index < fullName.length) {
        setDisplayedName((prev) => fullName.slice(0, index + 1))
        index++
      } else {
        clearInterval(interval)
        // Use setTimeout to ensure state update happens after typing completes
        setTimeout(() => setNameTypingDone(true), 100)
      }
    }, 100)
    
    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only run once on mount

  // Role rotation - runs once on mount
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentRoleIndex((prev) => (prev + 1) % roles.length)
    }, 2000)
    
    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only run once on mount

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("animate-fade-in-up")
            entry.target.classList.remove("opacity-0")
            setActiveSection(entry.target.id)
          }
        })
      },
      { threshold: 0.3, rootMargin: "0px 0px -20% 0px" },
    )

    sectionsRef.current.forEach((section) => {
      if (section) observer.observe(section)
    })

    return () => observer.disconnect()
  }, [])

  // ESC key to close modals
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelectedSkill(null)
        setSelectedExperience(null)
        setSelectedProject(null)
        setSelectedResearch(null)
      }
    }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [])

  const toggleLoadMore = (category: string) => {
    setExpandedSkills((prev) => ({
      ...prev,
      [category]: !prev[category],
    }))
  }

  // Icon mapping for skills - using actual tech brand logos
  const getSkillIcon = (skillName: string) => {
    const iconMap: { [key: string]: any } = {
      // Languages
      "Python": SiPython,
      "JavaScript": SiJavascript,
      "TypeScript": SiTypescript,
      "Java": Terminal, // Java doesn't have a simple-icons version, using Terminal as fallback
      "SQL": SiMysql,
      "HTML/CSS": SiHtml5,
      
      // Development & Databases
      "FastAPI": SiFastapi,
      "React.js": SiReact,
      "Node.js": SiNodedotjs,
      "PostgreSQL": SiPostgresql,
      "MongoDB": SiMongodb,
      "MySQL": SiMysql,
      "Spring Boot": SiSpringboot,
      "Flask": SiFlask,
      "Django": SiDjango,
      "Express": SiExpress,
      "Next.js": SiNextdotjs,
      "Vite": SiVite,
      
      // Cloud & DevOps
      "AWS": SiAmazon,
      "Google Cloud Platform": SiGooglecloud,
      "Azure": Terminal, // Using Terminal as fallback
      "Docker": SiDocker,
      "Kubernetes": SiKubernetes,
      "GitHub Actions": SiGithubactions,
      "Jenkins": SiJenkins,
      "MLflow": SiMlflow,
      "Kubeflow": SiKubernetes,
      
      // ML & AI
      "TensorFlow": SiTensorflow,
      "PyTorch": SiPytorch,
      "scikit-learn": SiScikitlearn,
      "XGBoost": Terminal,
      "LightGBM": Terminal,
      "Pandas": SiPandas,
      "NumPy": SiNumpy,
      "Hugging Face Transformers": SiHuggingface,
      "NLTK": Terminal,
      "spaCy": Terminal,
      "OpenCV": SiOpencv,
      "Computer Vision": SiOpencv,
      "Time-Series Forecasting": Terminal,
      "RAG (Retrieval-Augmented Generation)": Terminal,
      "LangChain": SiLangchain,
      "Ensemble Modeling": Terminal,
      "LSTM": Terminal,
      "CNN": Terminal,
    }
    
    return iconMap[skillName] || Terminal
  }

  const aboutTabs = {
    "outside-work": {
      title: "Outside Work",
      subtitle: "Community & Leadership Beyond Code",
      content:
        "Beyond my professional roles, I'm passionate about contributing to the tech community. I mentor aspiring data scientists and engineers, helping them navigate the intersection of machine learning and software engineering. I actively participate in tech meetups and conferences, sharing insights on scalable ML systems and cloud infrastructure.",
    },
    sde: {
      title: "Software Development Engineering",
      subtitle: "Building Scalable, Robust Systems",
      content:
        "With experience at Softvan and Project DAWN, I've built REST APIs, microservices, and full-stack applications. I'm proficient in Java, JavaScript, and Python, specializing in creating performant systems that handle high throughput. My focus is on clean architecture, security best practices, and building systems that scale.",
    },
    "data-science": {
      title: "Data Science & Machine Learning",
      subtitle: "From Data to Insights to Production",
      content:
        "My data science journey spans ensemble modeling, time-series forecasting, and production ML systems. I've built models achieving 87% accuracy for predictive maintenance and 85% for agricultural forecasting. I'm experienced in the entire ML lifecycle—from feature engineering and model development to deployment and monitoring using MLOps frameworks.",
    },
    leadership: {
      title: "Leadership & Mentorship",
      subtitle: "Guiding Teams & Growing Talent",
      content:
        "Throughout my internships and full-time roles, I've demonstrated leadership in technical decision-making and cross-functional collaboration. I design solutions that balance technical excellence with business requirements, mentor junior developers, and drive initiatives that improve team productivity and code quality.",
    },
  }

  const skillsData = [
    {
      category: "Languages",
      skills: [
        {
          name: "Python",
          proficiency: 95,
          description: "My primary language for data science, backend development, and ML engineering. Extensive experience with async programming, scientific computing, and production systems. Proficient in writing clean, maintainable code following PEP 8 standards and implementing design patterns for scalable applications.",
          usedIn: "Built LSTM/XGBoost ensemble models for Tusker AI achieving 87% HVAC failure prediction accuracy. Developed high-performance FastAPI microservices for Project DAWN processing terabyte-scale IoT data with sub-200ms response times. Engineered ETL pipelines reducing data prep time by 70%. Created NL-to-SQL conversion engine (QueryBridge AI) with 95% accuracy using LangChain and LLaMA.",
        },
        {
          name: "JavaScript",
          proficiency: 85,
          description: "Proficient in modern ES6+ JavaScript for building interactive, performant web applications. Strong understanding of asynchronous programming, closures, and event-driven architecture. Experienced with DOM manipulation, API integration, and state management in complex single-page applications.",
          usedIn: "Built Fatum AI SaaS platform processing 1K+ daily job listings from LinkedIn and Indeed with real-time updates. Developed RegNav AI chat interface using Vite for Maryland Department of Agriculture with WebSocket support for streaming LLM responses. Implemented client-side caching strategies reducing API calls by 40%.",
        },
        {
          name: "TypeScript",
          proficiency: 85,
          description: "Expert in leveraging TypeScript's static typing system to build robust, maintainable frontend and backend applications. Deep knowledge of advanced types, generics, and type inference. Experienced in setting up comprehensive TypeScript configurations for large-scale projects with strict mode enabled.",
          usedIn: "Architected type-safe React applications with custom hooks and context providers. Built full-stack Next.js applications with end-to-end type safety from database to UI. Implemented shared type definitions across microservices ensuring API contract consistency. Created reusable TypeScript utility types for domain-specific validation.",
        },
        {
          name: "Java",
          proficiency: 80,
          description: "Solid foundation in object-oriented programming with Java for enterprise backend systems. Experienced with Spring ecosystem, dependency injection, and building RESTful microservices. Strong understanding of JVM internals, garbage collection, and performance optimization for high-throughput applications.",
          usedIn: "Developed 10+ core REST APIs at Softvan using Spring Boot handling 10K+ daily financial transactions. Implemented JWT authentication and role-based access control for FinTech platform. Optimized SQL queries and database connection pooling improving transaction throughput by 35%. Built async event-driven microservices using Spring WebFlux.",
        },
        {
          name: "SQL",
          proficiency: 90,
          description: "Advanced SQL expertise including complex joins, window functions, CTEs, and query optimization. Deep understanding of database design, normalization, indexing strategies, and transaction management. Proficient in writing performant queries for large-scale data analysis and ETL operations.",
          usedIn: "Engineered terabyte-scale ETL pipelines for Project DAWN processing daily IoT sensor data from 500+ farms. Optimized complex analytical queries at Softvan reducing execution time from minutes to seconds using proper indexing and materialized views. Designed PostgreSQL schemas with pgvector for RegNav AI's RAG system supporting semantic search over thousands of regulatory documents.",
        },
        {
          name: "HTML/CSS",
          proficiency: 85,
          description: "Expert in semantic HTML5 and modern CSS3 including Flexbox, Grid, animations, and responsive design. Strong understanding of accessibility standards (WCAG), cross-browser compatibility, and CSS architecture patterns. Proficient with CSS preprocessors and modern utility-first frameworks like Tailwind CSS.",
          usedIn: "Built responsive, accessible interfaces for Fatum AI with mobile-first design supporting multiple devices. Implemented custom CSS animations and transitions for RegNav AI chat interface enhancing user experience. Created reusable component libraries with consistent design systems. Optimized CSS delivery reducing initial page load by 30%.",
        },
      ],
    },
    {
      category: "Development & Databases",
      skills: [
        {
          name: "FastAPI",
          proficiency: 90,
          description: "Expert in building high-performance async REST APIs with automatic OpenAPI documentation and Pydantic validation. Deep knowledge of dependency injection, middleware, background tasks, and WebSocket support. Experienced in optimizing API performance and implementing rate limiting, caching strategies.",
          usedIn: "Architected Project DAWN's core API delivering sub-200ms irrigation forecasts to 500+ farmers with 99.9% uptime. Built Tusker AI's ML microservices with async batch prediction endpoints. Developed QueryBridge AI's NL-to-SQL API with schema-aware prompting and query validation. Implemented RegNav AI's RAG pipeline with streaming LLM responses and pgvector integration.",
        },
        {
          name: "React.js",
          proficiency: 85,
          description: "Proficient in building complex, performant UIs using React hooks, context, and modern patterns. Strong understanding of component lifecycle, reconciliation algorithm, and performance optimization techniques. Experienced with state management libraries (Redux, Zustand) and React Server Components.",
          usedIn: "Built Fatum AI's dashboard processing 1K+ daily job listings with real-time updates and infinite scroll. Created RegNav AI's conversational interface with markdown rendering, code highlighting, and citation tracking. Developed Project DAWN's farmer-facing UI with interactive charts using Recharts. Implemented custom hooks for API integration and data fetching with error boundaries.",
        },
        {
          name: "Node.js",
          proficiency: 80,
          description: "Solid experience building scalable server-side applications using Node.js event-driven architecture. Proficient with Express.js, middleware patterns, and stream processing. Understanding of cluster mode, worker threads for CPU-intensive tasks, and memory management in production environments.",
          usedIn: "Developed backend services for SaaS platforms with JWT authentication and session management. Built WebSocket servers for real-time data synchronization. Implemented file upload/processing pipelines handling large datasets. Created CLI tools for automation and DevOps workflows with proper error handling and logging.",
        },
        {
          name: "PostgreSQL",
          proficiency: 90,
          description: "Advanced PostgreSQL expertise including JSONB operations, full-text search, window functions, and pgvector for semantic similarity. Deep understanding of query planning, vacuum operations, and replication strategies. Experienced with connection pooling, partitioning, and high-availability configurations.",
          usedIn: "Engineered Project DAWN's data warehouse processing terabytes of IoT sensor data with time-series optimizations and partitioning. Implemented RegNav AI's vector database using pgvector for semantic document retrieval with 99% recall. Designed Tusker AI's telemetry database with proper indexing reducing query time by 60%. Built Softvan's transactional database handling 10K+ daily operations with ACID guarantees.",
        },
        {
          name: "MongoDB",
          proficiency: 80,
          description: "Proficient in NoSQL database design with MongoDB including aggregation pipelines, indexing strategies, and sharding. Understanding of document modeling, replica sets for high availability, and change streams for real-time data synchronization. Experienced with MongoDB Atlas cloud deployment.",
          usedIn: "Designed Fatum AI's multi-tenant schema with role-based access control supporting 100+ concurrent users. Implemented flexible document structures for job listing data with embedded references. Built aggregation pipelines for analytics dashboards. Optimized compound indexes reducing query latency by 50%.",
        },
        {
          name: "MySQL",
          proficiency: 85,
          description: "Strong foundation in MySQL database administration, query optimization, and InnoDB storage engine internals. Experienced with master-slave replication, backup strategies, and performance tuning using EXPLAIN plans. Proficient in stored procedures, triggers, and transaction isolation levels.",
          usedIn: "Managed production databases for enterprise applications with automated backup systems. Optimized complex JOIN queries for reporting systems reducing execution time by 70%. Implemented database migration strategies with zero downtime. Configured read replicas for load distribution across analytical and transactional workloads.",
        },
        {
          name: "Spring Boot",
          proficiency: 85,
          description: "Expert in Spring Boot ecosystem including Spring MVC, Spring Data JPA, Spring Security, and Spring Cloud. Deep knowledge of dependency injection, AOP, and microservices patterns. Experienced in building production-grade applications with actuator endpoints, distributed tracing, and circuit breakers.",
          usedIn: "Built Softvan's core FinTech APIs handling 10K+ daily transactions with comprehensive error handling and audit logging. Implemented OAuth2 resource servers with JWT validation. Configured Hystrix circuit breakers for resilient microservices. Developed async messaging using Spring AMQP with RabbitMQ for order processing workflows.",
        },
        {
          name: "Flask",
          proficiency: 80,
          description: "Proficient in Flask for building lightweight Python web applications and RESTful APIs. Understanding of Blueprints for modular architecture, custom decorators, and middleware. Experienced with SQLAlchemy ORM, Flask-RESTful, and deployment strategies for production environments.",
          usedIn: "Rapid prototyping of ML model serving endpoints with custom validation and error handling. Built internal tools and admin dashboards with Flask-Admin. Developed microservices for data processing pipelines. Implemented API authentication using Flask-JWT and rate limiting with Flask-Limiter.",
        },
        {
          name: "Django",
          proficiency: 75,
          description: "Solid understanding of Django's MTV architecture, ORM, and admin interface. Experienced with Django REST Framework for building APIs, Celery for async tasks, and Django channels for WebSocket support. Knowledge of security best practices and middleware implementation.",
          usedIn: "Built content management systems with Django admin customizations. Developed REST APIs with serializers and viewsets. Implemented background job processing with Celery for email notifications and data exports. Created custom middleware for request logging and authentication.",
        },
        {
          name: "Express",
          proficiency: 80,
          description: "Proficient in Express.js for building scalable Node.js APIs with middleware-based architecture. Understanding of routing, error handling, and request/response lifecycle. Experienced with template engines, session management, and security best practices using Helmet.js.",
          usedIn: "Developed RESTful APIs with comprehensive validation using express-validator. Implemented authentication middleware with Passport.js supporting multiple strategies. Built file upload endpoints with Multer for image processing. Created custom middleware for logging, CORS, and rate limiting.",
        },
        {
          name: "Next.js",
          proficiency: 85,
          description: "Advanced Next.js expertise including App Router, Server Components, Server Actions, and streaming SSR. Deep understanding of static site generation, incremental static regeneration, and API routes. Experienced with image optimization, font loading strategies, and deployment on Vercel.",
          usedIn: "Built performant full-stack applications with file-based routing and automatic code splitting. Implemented hybrid rendering strategies combining SSR and SSG for optimal performance. Created API routes with middleware for authentication and data validation. Leveraged React Server Components for reduced client-side JavaScript bundle sizes.",
        },
        {
          name: "Vite",
          proficiency: 85,
          description: "Expert in Vite's modern build tooling with instant hot module replacement and optimized production builds. Understanding of ES modules, plugin architecture, and development server configuration. Experienced with framework-specific templates and build optimization strategies.",
          usedIn: "Configured RegNav AI's frontend with Vite for blazing-fast development experience and 3x faster build times. Implemented custom plugins for environment variable handling and asset optimization. Optimized chunk splitting strategies reducing initial load time by 40%. Configured proxy for API development with hot reload.",
        },
      ],
    },
    {
      category: "Cloud & DevOps",
      skills: [
        {
          name: "AWS (EC2, S3, Lambda, RDS)",
          proficiency: 90,
          description: "Extensive experience with AWS cloud services including EC2, S3, RDS, Lambda, and ECS. Proficient in designing serverless architectures, implementing auto-scaling, and cost optimization. Strong understanding of IAM, VPC networking, CloudWatch monitoring, and infrastructure as code using CloudFormation or CDK.",
          usedIn: "Deployed Project DAWN on AWS with EC2 auto-scaling handling variable farmer traffic loads. Configured S3 for IoT data lake storage with lifecycle policies. Set up RDS PostgreSQL with Multi-AZ deployment and automated backups. Implemented Lambda functions for data processing triggers and API Gateway for serverless endpoints. Used CloudWatch for monitoring and alerting.",
        },
        {
          name: "Docker",
          proficiency: 85,
          description: "Expert in containerization using Docker for consistent development and production environments. Deep understanding of Dockerfile optimization, multi-stage builds, layer caching, and image security scanning. Proficient with Docker Compose for local development and Docker networking.",
          usedIn: "Containerized all microservices for Tusker AI with multi-stage builds reducing image sizes by 60%. Created Docker Compose configurations for local development mimicking production. Implemented health checks and graceful shutdown handlers. Built CI pipelines for automated image building, scanning, and registry push. Optimized layer caching for faster builds.",
        },
        {
          name: "Kubernetes",
          proficiency: 85,
          description: "Advanced Kubernetes expertise including deployments, services, ingress, config maps, secrets, and persistent volumes. Understanding of pod scheduling, resource limits, horizontal pod autoscaling, and rolling updates. Experienced with Helm charts, kubectl, and monitoring with Prometheus/Grafana.",
          usedIn: "Orchestrated Project DAWN microservices on Kubernetes with HPA handling 500+ concurrent farmer requests. Configured ingress controllers with SSL termination and load balancing. Implemented ConfigMaps for environment-specific configurations and Secrets for credential management. Set up persistent volumes for database stateful sets. Deployed Prometheus for metrics collection and Grafana dashboards.",
        },
        {
          name: "GitHub Actions",
          proficiency: 80,
          description: "Proficient in building comprehensive CI/CD pipelines with GitHub Actions including workflow automation, matrix builds, caching strategies, and deployment to multiple environments. Understanding of workflow syntax, secrets management, and action marketplace for reusable components.",
          usedIn: "Built automated CI/CD pipelines for all projects with linting, testing, building, and deployment stages. Implemented matrix builds for testing across multiple Python/Node versions. Configured automated Docker image building and pushing to container registries. Set up deployment workflows with environment protection rules and manual approvals. Created custom composite actions for reusable workflow components.",
        },
        {
          name: "GitLab CI/CD",
          proficiency: 80,
          description: "Solid experience with GitLab CI/CD for automated testing and deployment pipelines. Understanding of .gitlab-ci.yml configuration, pipeline stages, artifacts, and environment-specific deployments. Experienced with GitLab Runner configuration and integration with container registries.",
          usedIn: "Configured enterprise deployment pipelines with automated testing, security scanning, and multi-stage deployments. Implemented pipeline optimization with caching and parallelization reducing CI time by 50%. Set up auto-scaling runners for cost-effective build capacity. Integrated with external tools for notifications and deployment approvals.",
        },
        {
          name: "Jenkins",
          proficiency: 75,
          description: "Solid experience with Jenkins for continuous integration and deployment automation. Understanding of pipeline as code using Jenkinsfiles, build agents configuration, and plugin ecosystem. Experienced with distributed builds, parallel execution, and integration with version control systems.",
          usedIn: "Configured Jenkins pipelines at Softvan for automated testing and deployment of Spring Boot microservices. Set up multi-branch pipelines for feature branch testing. Implemented parallel test execution reducing CI time by 40%. Integrated with SonarQube for code quality gates. Configured build agents with Docker for isolated build environments.",
        },
        {
          name: "MLflow",
          proficiency: 85,
          description: "Expert in MLflow for end-to-end machine learning lifecycle management including experiment tracking, model registry, and deployment. Understanding of MLflow Projects for reproducible runs, MLflow Models for packaging, and MLflow Tracking for logging metrics and parameters.",
          usedIn: "Tracked over 500+ ML experiments for Tusker AI with hyperparameter logging and model comparison. Registered production models with versioning and stage transitions. Deployed models as REST endpoints using MLflow serve. Implemented automated retraining pipelines with MLflow integration. Created custom MLflow plugins for team-specific metrics.",
        },
        {
          name: "Kubeflow",
          proficiency: 80,
          description: "Proficient in Kubeflow for building and deploying ML workflows on Kubernetes. Understanding of Kubeflow Pipelines for orchestration, KFServing for model serving, and Katib for hyperparameter tuning. Experienced with notebook servers and multi-user isolation.",
          usedIn: "Built ML pipeline orchestration for ensemble model training with parallel component execution. Configured KFServing for scalable model inference endpoints with canary deployments. Used Katib for automated hyperparameter optimization across distributed GPUs. Implemented multi-step pipelines for data preprocessing, training, and evaluation with artifact tracking.",
        },
      ],
    },
    {
      category: "Machine Learning & AI",
      skills: [
        {
          name: "TensorFlow",
          proficiency: 85,
          description: "Expert in TensorFlow 2.x for building production-grade deep learning models. Proficient with Keras API, custom training loops, TensorFlow Extended (TFX) for ML pipelines, and TensorFlow Serving for model deployment. Understanding of distributed training, mixed-precision training, and model optimization.",
          usedIn: "Built CNN architectures for computer vision projects with transfer learning from pre-trained models. Implemented custom training loops with tf.GradientTape for complex loss functions. Deployed models using TensorFlow Serving with REST and gRPC endpoints. Optimized models with quantization and pruning reducing inference time by 40%.",
        },
        {
          name: "PyTorch",
          proficiency: 85,
          description: "Advanced PyTorch expertise for research and production deep learning. Strong understanding of autograd, dynamic computation graphs, and custom layer implementation. Experienced with PyTorch Lightning for cleaner code organization, TorchServe for deployment, and ONNX for model export.",
          usedIn: "Developed LSTM networks for Tusker AI time-series HVAC failure prediction achieving 87% accuracy. Built transformer-based models for NLP tasks with custom attention mechanisms. Implemented ensemble training with early stopping and learning rate scheduling. Used PyTorch Lightning for distributed training across multiple GPUs.",
        },
        {
          name: "scikit-learn",
          proficiency: 90,
          description: "Comprehensive scikit-learn mastery including all major algorithms, model selection techniques, pipeline construction, and custom transformers. Deep understanding of cross-validation strategies, hyperparameter tuning with GridSearchCV/RandomizedSearchCV, and ensemble methods for robust predictions.",
          usedIn: "Built ensemble models combining Random Forests, Gradient Boosting, and SVMs for Project DAWN irrigation forecasting. Implemented custom feature transformers for agricultural data preprocessing. Used StackingClassifier for meta-learning achieving 85% prediction accuracy. Performed extensive feature engineering with SelectKBest and PCA for dimensionality reduction.",
        },
        {
          name: "XGBoost",
          proficiency: 90,
          description: "Expert in XGBoost gradient boosting for high-performance predictions. Deep understanding of tree-based learning, regularization parameters, early stopping, and handling imbalanced datasets. Proficient in hyperparameter tuning using Bayesian optimization and feature importance analysis for interpretability.",
          usedIn: "Achieved 87% accuracy for Tusker AI HVAC failure prediction 2-3 days in advance using XGBoost with custom objective functions. Built credit risk models for AmEx dataset with class weight balancing. Performed extensive hyperparameter tuning with Optuna finding optimal tree depth, learning rate, and subsample ratios. Analyzed SHAP values for model interpretability.",
        },
        {
          name: "LightGBM",
          proficiency: 90,
          description: "Advanced LightGBM expertise for fast, distributed gradient boosting on large datasets. Understanding of leaf-wise tree growth, categorical feature handling, and GPU acceleration. Experienced with dart booster for improved generalization and handling of high-cardinality features.",
          usedIn: "Built ensemble models for AmEx credit prediction processing millions of transactions with LightGBM's fast training. Leveraged categorical feature support avoiding one-hot encoding overhead. Implemented cross-validation with early stopping preventing overfitting. Used feature interaction constraints for better model interpretability.",
        },
        {
          name: "Pandas",
          proficiency: 95,
          description: "Expert-level Pandas proficiency for data manipulation and analysis. Mastery of DataFrame operations, groupby aggregations, merge/join strategies, time-series analysis, and performance optimization with vectorization. Understanding of memory management and chunk processing for large datasets.",
          usedIn: "Processed terabyte-scale IoT data for Project DAWN using optimized Pandas operations with chunking and categorical dtypes. Performed complex multi-index operations for hierarchical data analysis. Built ETL pipelines with method chaining reducing data prep time by 70%. Implemented efficient time-series resampling and rolling window calculations for agricultural forecasting.",
        },
        {
          name: "NumPy",
          proficiency: 90,
          description: "Advanced NumPy expertise for numerical computing and array operations. Deep understanding of broadcasting, vectorization, linear algebra operations, and FFT. Proficient in memory-efficient array manipulation and integration with C/Fortran code for performance-critical operations.",
          usedIn: "Implemented Monte Carlo simulations for financial modeling using vectorized NumPy operations achieving 100x speedup over loops. Built custom loss functions and activation derivatives for neural networks. Performed large-scale matrix operations for dimensionality reduction algorithms. Optimized memory usage with dtype selection and in-place operations.",
        },
        {
          name: "Hugging Face Transformers",
          proficiency: 85,
          description: "Proficient with Hugging Face ecosystem for state-of-the-art NLP and vision models. Experienced with pre-trained models (BERT, GPT, T5), fine-tuning strategies, tokenization, and model deployment. Understanding of attention mechanisms and transfer learning best practices.",
          usedIn: "Fine-tuned BERT models for RegNav AI document classification and semantic search. Implemented GPT-based text generation for Fatum AI resume tailoring improving relevance by 85%. Used sentence transformers for embedding generation in RAG pipelines. Deployed models with optimized inference using ONNX Runtime.",
        },
        {
          name: "NLTK",
          proficiency: 80,
          description: "Natural language toolkit for text processing and analysis",
          usedIn: "Natural language processing, text analysis",
        },
        {
          name: "spaCy",
          proficiency: 85,
          description: "Industrial-strength NLP library for production-ready text processing",
          usedIn: "RegNav AI entity extraction, NLP pipelines",
        },
        {
          name: "OpenCV",
          proficiency: 80,
          description: "Computer vision library for image and video processing",
          usedIn: "CNN architectures, image classification",
        },
        {
          name: "Computer Vision",
          proficiency: 80,
          description: "Field of AI focused on enabling computers to understand and analyze images",
          usedIn: "CNN architectures, image classification",
        },
        {
          name: "Time-Series Forecasting",
          proficiency: 90,
          description: "Techniques for predicting future values based on historical temporal data",
          usedIn: "Project DAWN irrigation predictions, HVAC failure forecasting",
        },
        {
          name: "RAG (Retrieval-Augmented Generation)",
          proficiency: 85,
          description: "Technique combining information retrieval with generative models for accurate responses",
          usedIn: "RegNav AI legal chatbot, LLM-powered search",
        },
        {
          name: "LLMs & OpenAI API",
          proficiency: 90,
          description: "Large language models and APIs for natural language understanding and generation",
          usedIn: "Fatum AI resume generation, QueryBridge NL-to-SQL, RegNav conversational AI",
        },
        {
          name: "LangChain",
          proficiency: 85,
          description: "Framework for building applications with language models and external tools",
          usedIn: "RegNav AI, QueryBridge, LLM chain orchestration",
        },
        {
          name: "PySpark",
          proficiency: 85,
          description: "Python API for Apache Spark enabling large-scale distributed data processing",
          usedIn: "Project DAWN 70% faster ETL pipelines, terabyte-scale data processing",
        },
        {
          name: "BERT",
          proficiency: 85,
          description: "Pre-trained transformer model for bidirectional language understanding",
          usedIn: "Transfer learning, semantic search, text classification",
        },
        {
          name: "GPT",
          proficiency: 90,
          description: "Generative pre-trained transformer models for diverse language tasks",
          usedIn: "Content generation, code assistance, conversational AI",
        },
        {
          name: "LSTMs",
          proficiency: 85,
          description: "Recurrent neural networks designed for sequence modeling and time-series",
          usedIn: "Tusker AI HVAC predictions, temporal pattern recognition",
        },
        {
          name: "CNNs",
          proficiency: 80,
          description: "Convolutional neural networks for image recognition and feature extraction",
          usedIn: "Computer vision, image classification, feature learning",
        },
      ],
    },
  ]

  const experienceData = [
    {
      year: "2024",
      role: "Full Stack Developer",
      company: "Project DAWN, University of Maryland",
      location: "College Park, MD",
      duration: "Jan 2024 - Present",
      description: "Built predictive irrigation platform with sub-200ms forecasts serving 500+ farmers. Engineered ETL pipelines reducing data prep by 70%.",
      achievements: [
        "Developed real-time ML prediction system with sub-200ms response time serving 500+ farmers",
        "Built ETL pipelines processing weather/soil data, reducing manual prep time by 70%",
        "Designed React dashboard with interactive visualizations for irrigation recommendations",
        "Deployed scalable microservices on AWS with Kubernetes orchestration"
      ],
      tech: ["FastAPI", "React", "AWS", "PostgreSQL", "Kubernetes", "Docker", "Redis"],
    },
    {
      year: "2023",
      role: "Data Science Intern",
      company: "Tusker AI",
      location: "Remote",
      duration: "Jun 2023 - Dec 2023",
      description: "Developed ensemble LSTM/XGBoost models with 87% accuracy for HVAC failure prediction. Deployed ML microservices reducing costs by 18%.",
      achievements: [
        "Built ensemble LSTM/XGBoost models achieving 87% accuracy in HVAC failure prediction",
        "Deployed production-ready ML microservices using FastAPI and Docker",
        "Reduced operational costs by 18% through predictive maintenance insights",
        "Implemented automated model retraining pipeline with MLflow tracking"
      ],
      tech: ["Python", "TensorFlow", "XGBoost", "FastAPI", "PostgreSQL", "Docker", "MLflow"],
    },
    {
      year: "2022",
      role: "Software Development Intern",
      company: "Softvan Pvt Ltd",
      location: "Ahmedabad, India",
      duration: "May 2022 - Aug 2022",
      description: "Built 10+ REST APIs using Spring Boot handling 10K+ daily transactions. Optimized SQL queries for large-scale operations.",
      achievements: [
        "Developed 10+ RESTful APIs using Spring Boot handling 10K+ daily transactions",
        "Optimized complex SQL queries improving response time by 40%",
        "Implemented JWT authentication and role-based access control",
        "Collaborated in agile team using Git, JIRA, and CI/CD pipelines"
      ],
      tech: ["Java", "Spring Boot", "PostgreSQL", "Redis", "Docker", "Git"],
    },
  ]

  const projectsData = [
    {
      title: "Credit Card Default Prediction",
      description:
        "End-to-end credit risk scoring pipeline processing 50M+ American Express transactions, achieving top 5% global ranking",
      keyMetrics: "Top 5% global ranking • 50M+ transactions • Hybrid ensemble",
      tech: ["XGBoost", "LightGBM", "Transformers", "Kubeflow", "MLflow"],
      achievements: [
        "Designed hybrid sequential ensemble (XGBoost, LightGBM, Transformers) with cost-aware loss function",
        "Established governed MLOps framework for automated model validation and feature lineage tracking",
        "Optimized financial loss minimization through sophisticated ensemble stacking",
      ],
      demoUrl: "#", // Update later
      githubUrl: "#", // Update later
      videoUrl: "/Videos/Screen Recording 2025-11-16 at 5.25.52 PM.mov",
    },
    {
      title: "Fatum AI - Automated Job Co-Pilot Platform",
      description:
        "Full-stack SaaS application automating job search workflows with intelligent resume and cover letter generation",
      keyMetrics: "1K+ job listings daily • 85% application relevance improvement • Multi-tenant",
      tech: ["React.js", "FastAPI", "MongoDB", "OpenAI API", "Docker"],
      achievements: [
        "Built full-stack SaaS processing 1K+ job listings daily from LinkedIn and Indeed",
        "Integrated OpenAI API for intelligent resume tailoring improving relevance by 85%",
        "Designed MongoDB schema with role-based access control for scalable multi-tenant deployment",
      ],
      demoUrl: "#", // Update later
      githubUrl: "#", // Update later
      videoUrl: "/Videos/Fatum.mov",
    },
    {
      title: "RegNav AI - Legal/Regulatory Conversational AI",
      description:
        "Maryland Department of Agriculture legal chatbot with RAG + LLM architecture, enabling natural language queries over regulatory documents",
      keyMetrics: "LLM-assisted retrieval • Structured citations • High concurrent users",
      tech: ["FastAPI", "Vite", "Supabase", "pgvector", "LangChain", "LLama"],
      achievements: [
        "Built end-to-end RAG system with LLM-assisted query rewrite and re-ranking",
        "Productionized with tiny DB pooling, auth, chat history, and optimized ingestion for thousands of docs",
        "Implemented cross-references and Title 15/26-only web fallback for compliance",
      ],
      demoUrl: "#", // Update later
      githubUrl: "#", // Update later
      videoUrl: "/Videos/Regnav.mov",
    },
    {
      title: "QueryBridge AI - Natural Language to SQL Conversion",
      description:
        "Semantic NL-to-SQL engine enabling non-technical stakeholders to safely query enterprise financial databases",
      keyMetrics: "95% accuracy • Schema-aware • SQL injection prevention",
      tech: ["FastAPI", "LangChain", "LLaMA API", "PostgreSQL"],
      achievements: [
        "Engineered semantic NL-to-SQL engine achieving 95% accuracy for enterprise queries",
        "Deployed as secure FastAPI microservice with input validation, caching, and 60% query latency reduction",
        "Implemented schema-aware prompting framework for controlled, auditable data access",
      ],
      demoUrl: "#", // Update later
      githubUrl: "#", // Update later
      videoUrl: "/Videos/Querybridge.mov",
    },
  ]

  const researchData = [
    {
      title: "Transfer Learning-based Emotion Detection System in Cultivating Workplace Harmony",
      publisher: "IEEE Xplore",
      year: "2024",
      description:
        "Research on applying transfer learning techniques to emotion detection systems for improving workplace dynamics and team collaboration.",
      url: "https://ieeexplore.ieee.org/document/10475202",
      abstract: "This paper presents a novel approach to workplace emotion detection using transfer learning techniques. We demonstrate how pre-trained models can be fine-tuned to recognize emotional states in professional settings, leading to improved team dynamics and workplace harmony.",
      keywords: ["Transfer Learning", "Emotion Detection", "Workplace Dynamics", "Deep Learning"],
    },
    {
      title: "Hybrid Convolutional Neural Mixed Approached Model for Incorporating Sign Language Features",
      publisher: "Springer Nature",
      year: "2023",
      description:
        "Development of hybrid CNN architecture combining multiple approaches for accurate sign language recognition and feature extraction.",
      url: "https://link.springer.com/article/10.1007/s42979-025-04131-w?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20250623&utm_content=10.1007/s42979-025-04131-w",
      abstract: "We propose a hybrid convolutional neural network architecture that combines spatial and temporal features for enhanced sign language recognition. The model achieves state-of-the-art performance on standard benchmarks.",
      keywords: ["CNN", "Sign Language Recognition", "Computer Vision", "Feature Extraction"],
    },
    {
      title: "Intelligent Data Dissemination in Vehicular Networks: Leveraging Reinforcement Learning",
      publisher: "Springer Nature",
      year: "2023",
      description:
        "Application of reinforcement learning techniques to optimize data distribution strategies in vehicular networks for improved communication efficiency.",
      url: "https://link.springer.com/chapter/10.1007/978-981-96-5190-0_9",
      abstract: "This research explores the application of reinforcement learning algorithms to optimize data dissemination strategies in vehicular ad-hoc networks (VANETs). Our approach significantly improves communication efficiency and reduces latency.",
      keywords: ["Reinforcement Learning", "Vehicular Networks", "Data Dissemination", "V2V Communication"],
    },
  ]

  const sectionsRef = useRef<(HTMLElement | null)[]>([])

  return (
    <div className="min-h-screen bg-background text-foreground relative dark">
      <nav className="fixed left-8 top-1/2 -translate-y-1/2 z-10 hidden lg:block">
        <div className="flex flex-col gap-4">
          {["intro", "about", "skills", "work", "projects", "research", "connect"].map((section) => (
            <button
              key={section}
              onClick={() => document.getElementById(section)?.scrollIntoView({ behavior: "smooth" })}
              className={`w-2 h-8 rounded-full transition-all duration-500 ${
                activeSection === section ? "bg-foreground" : "bg-muted-foreground/30 hover:bg-muted-foreground/60"
              }`}
              aria-label={`Navigate to ${section}`}
            />
          ))}
        </div>
      </nav>

      <ShaderBackground>
        {/* Top Navigation Bar */}
        <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-sm bg-background/10 border-b border-white/5">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between md:justify-center h-14 sm:h-16">
              {/* Logo - Mobile only */}
              <Link href="#intro" className="text-lg font-light text-white/70 md:hidden">
                PS
              </Link>

              {/* Desktop Navigation Links - Centered */}
              <div className="hidden md:flex items-center justify-center gap-6 sm:gap-8 md:gap-10 lg:gap-12">
                {[
                  { name: "About", id: "about" },
                  { name: "Skills", id: "skills" },
                  { name: "Work", id: "work" },
                  { name: "Projects", id: "projects" },
                  { name: "Connect", id: "connect" },
                ].map((item) => (
                  <button
                    key={item.id}
                    onClick={() => document.getElementById(item.id)?.scrollIntoView({ behavior: "smooth" })}
                    className="text-sm text-white/60 hover:text-white/90 transition-colors duration-300 font-light"
                  >
                    {item.name}
                  </button>
                ))}
              </div>

              {/* Hamburger Menu Button - Mobile only */}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden text-white/60 hover:text-white/90 transition-colors p-2"
                aria-label="Toggle menu"
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  {mobileMenuOpen ? (
                    <path d="M6 18L18 6M6 6l12 12" />
                  ) : (
                    <path d="M4 6h16M4 12h16M4 18h16" />
                  )}
                </svg>
              </button>
            </div>
          </div>

          {/* Mobile Menu Dropdown */}
          <div
            className={`md:hidden backdrop-blur-md bg-background/20 border-t border-white/5 transition-all duration-300 ease-in-out overflow-hidden ${
              mobileMenuOpen ? "max-h-screen opacity-100" : "max-h-0 opacity-0"
            }`}
          >
            <div className="px-4 py-4 space-y-3">
              {[
                { name: "About", id: "about" },
                { name: "Skills", id: "skills" },
                { name: "Work", id: "work" },
                { name: "Projects", id: "projects" },
                { name: "Connect", id: "connect" },
              ].map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    document.getElementById(item.id)?.scrollIntoView({ behavior: "smooth" })
                    setMobileMenuOpen(false)
                  }}
                  className="block w-full text-left py-2 px-3 text-sm text-white/60 hover:text-white/90 hover:bg-white/5 rounded transition-colors duration-300 font-light"
                >
                  {item.name}
                </button>
              ))}
            </div>
          </div>
        </nav>

        <main className="max-w-4xl mx-auto px-6 sm:px-8 lg:px-16">
          <header
            id="intro"
            ref={(el) => {
              if (el) sectionsRef.current[0] = el
            }}
            className="min-h-screen flex items-center opacity-0"
          >
            <div className="grid lg:grid-cols-5 gap-12 sm:gap-16 w-full relative z-10">
              <div className="lg:col-span-3 space-y-6 sm:space-y-8">
                <div className="space-y-3 sm:space-y-2">
                  <div className="h-6 sm:h-8 relative">
                    <div
                      className="text-sm text-white/70 font-mono tracking-wider"
                      style={{
                        opacity: nameTypingDone ? 1 : 0,
                        transform: nameTypingDone ? 'translateX(0)' : 'translateX(-32px)',
                        transition: 'opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1), transform 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
                      }}
                    >
                      {roles.map((role, idx) => (
                        <span
                          key={role}
                          className={`block absolute transition-opacity duration-500 ${
                            idx === currentRoleIndex ? "opacity-100" : "opacity-0"
                          }`}
                        >
                          {role.toUpperCase()}
                        </span>
                      ))}
                    </div>
                  </div>
                  <h1 className="text-5xl sm:text-6xl lg:text-7xl font-light tracking-tight text-white">
                    {displayedName.split(' ').map((word, idx) => (
                      <span key={idx} className={idx === 1 ? 'text-white/60' : ''}>
                        {word}{idx === 0 ? ' ' : ''}
                      </span>
                    ))}
                  </h1>
                </div>

                <div
                  className="space-y-6 max-w-md"
                  style={{
                    opacity: nameTypingDone ? 1 : 0,
                    transform: nameTypingDone ? 'translateX(0)' : 'translateX(-32px)',
                    transition: 'opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.2s, transform 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.2s',
                  }}
                >
                  <p className="text-lg sm:text-xl text-white/70 leading-relaxed">
                    Data wrangler and code architect fueled by caffeine and curiosity. Tame Impala for clean code
                    commits, Kanye for peak hustle. Daydreams included, free of charge.
                  </p>

                  <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4 text-sm text-white/70">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                      Open to work
                    </div>
                    <div>Washington DC</div>
                  </div>
                </div>
              </div>

              <div
                className="lg:col-span-2 flex flex-col justify-end space-y-6 sm:space-y-8 mt-8 lg:mt-0"
                style={{
                  opacity: nameTypingDone ? 1 : 0,
                  transform: nameTypingDone ? 'translateX(0)' : 'translateX(-32px)',
                  transition: 'opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.4s, transform 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.4s',
                }}
              >
                <div className="space-y-4">
                  <div className="text-sm text-white/70 font-mono">CURRENTLY</div>
                  <div className="space-y-2">
                    <div className="text-white">ML Engineer</div>
                    <div className="text-white/70">@ Department of Computer Science, UMD</div>
                    <div className="text-xs text-white/70">Present</div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="text-sm text-white/70 font-mono">FOCUS</div>
                  <div className="flex flex-wrap gap-2">
                    {["Python", "FastAPI", "React", "PostgreSQL", "AWS"].map((skill) => (
                      <span
                        key={skill}
                        className="px-3 py-1 text-xs border border-white/30 rounded-full text-white/70 hover:border-white/50 hover:text-white transition-all duration-300"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </header>
        </main>
      </ShaderBackground>

      <main className="max-w-4xl mx-auto px-6 sm:px-8 lg:px-16">
        {/* ABOUT SECTION - Second */}
        <section
          id="about"
          ref={(el) => {
            if (el) sectionsRef.current[1] = el
          }}
          className="min-h-screen py-20 sm:py-32"
        >
          <div className="space-y-12 sm:space-y-16">
            <div className="space-y-2">
              <h2 className="text-3xl sm:text-4xl font-light section-header-glow">
                About - Jack of All Trades, Master of a Few
              </h2>
              <p className="text-sm text-muted-foreground/60 font-mono tracking-wider">
                Discover my diverse expertise across domains
              </p>
            </div>

            <div className="flex flex-wrap gap-3 border-b border-border/30 pb-0">
              {Object.entries(aboutTabs).map(([key, tab]) => (
                <button
                  key={key}
                  onClick={() => setActiveAboutTab(key)}
                  className={`px-4 py-3 text-sm font-medium transition-all duration-300 relative group ${
                    activeAboutTab === key ? "text-foreground" : "text-muted-foreground hover:text-muted-foreground/80"
                  }`}
                >
                  {tab.title}
                  {activeAboutTab === key && (
                    <span className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-purple-500/60 via-purple-400/40 to-transparent animate-fade-in-up"></span>
                  )}
                </button>
              ))}
            </div>

            <div className="space-y-6 section-gradient-accent p-6 sm:p-8 rounded-lg">
              {Object.entries(aboutTabs).map(
                ([key, tab]) =>
                  activeAboutTab === key && (
                    <div key={key} className="space-y-6 animate-fade-in-up">
                      <div className="space-y-2">
                        <h3 className="text-2xl sm:text-3xl font-light">{tab.title}</h3>
                        <p className="text-lg text-muted-foreground/70 italic">{tab.subtitle}</p>
                      </div>
                      <p className="text-muted-foreground leading-relaxed max-w-3xl">{tab.content}</p>
                    </div>
                  ),
              )}
            </div>
          </div>
        </section>

        {/* SKILLS SECTION - Third */}
        <section
          id="skills"
          ref={(el) => {
            if (el) sectionsRef.current[2] = el
          }}
          className="min-h-screen py-20 sm:py-32"
        >
          <div className="space-y-12 sm:space-y-16">
            <div className="space-y-2">
              <h2 className="text-3xl sm:text-4xl font-light section-header-glow">Skills & Expertise</h2>
              <p className="text-sm text-muted-foreground/60 font-mono tracking-wider">
                Technical arsenal across domains - Click any skill to learn more
              </p>
            </div>

            <div className="flex flex-wrap gap-3 border-b border-border/30 pb-0">
              {skillsData.map((skillGroup) => (
                <button
                  key={skillGroup.category}
                  onClick={() => setActiveSkillsTab(skillGroup.category)}
                  className={`px-4 py-3 text-sm font-medium transition-all duration-300 relative group ${
                    activeSkillsTab === skillGroup.category
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-muted-foreground/80"
                  }`}
                >
                  {skillGroup.category}
                  {activeSkillsTab === skillGroup.category && (
                    <span className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-purple-500/60 via-purple-400/40 to-transparent animate-fade-in-up"></span>
                  )}
                </button>
              ))}
            </div>

            <div className="space-y-10 sm:space-y-14">
              {skillsData.map((skillGroup) =>
                activeSkillsTab === skillGroup.category ? (
                  <div key={skillGroup.category} className="space-y-5 animate-fade-in-up">
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2.5 sm:gap-3">
                      {skillGroup.skills.map(
                        (skill, skillIndex) =>
                          (skillIndex < 4 || expandedSkills[skillGroup.category]) && (
                            <button
                              key={`${skillGroup.category}-${skillIndex}`}
                              onClick={() => setSelectedSkill(skill)}
                              className="group relative p-3 sm:p-3.5 rounded-lg border border-border/60 bg-muted/20 hover-gradient-accent transition-all duration-300 text-left focus:outline-none focus:ring-2 focus:ring-ring/50"
                            >
                              <div className="space-y-2">
                                <div className="flex items-start justify-between gap-2">
                                  <div className="flex items-center gap-2 flex-1 min-w-0">
                                    {(() => {
                                      const IconComponent = getSkillIcon(skill.name)
                                      return (
                                        <IconComponent 
                                          className="w-4 h-4 sm:w-5 sm:h-5 text-muted-foreground/70 group-hover:text-foreground/80 transition-colors flex-shrink-0" 
                                        />
                                      )
                                    })()}
                                    <span className="text-xs sm:text-sm font-medium text-foreground group-hover:text-muted-foreground transition-colors line-clamp-2">
                                      {skill.name}
                                    </span>
                                  </div>
                                </div>
                                <div className="h-0.5 bg-border rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-gradient-to-r from-purple-500/60 via-purple-400/50 to-foreground/30 transition-all duration-500 group-hover:from-purple-500/80 group-hover:via-purple-400/70"
                                    style={{ width: `${skill.proficiency}%` }}
                                  ></div>
                                </div>
                                <div className="text-xs text-muted-foreground/60 font-mono">{skill.proficiency}%</div>
                              </div>
                            </button>
                          ),
                      )}
                    </div>
                    
                    {/* Detailed Skill Information Panel */}
                    {selectedSkill && (
                      <div className="mt-8 p-6 sm:p-8 rounded-xl border border-purple-500/30 bg-gradient-to-br from-purple-500/5 via-muted/20 to-transparent backdrop-blur-sm animate-fade-in-up">
                        <div className="space-y-6">
                          {/* Header with Icon and Name */}
                          <div className="flex items-start gap-4">
                            <div className="p-3 rounded-lg bg-muted/30 border border-border/40">
                              {(() => {
                                const IconComponent = getSkillIcon(selectedSkill.name)
                                return (
                                  <IconComponent className="w-8 h-8 text-foreground/80" />
                                )
                              })()}
                            </div>
                            <div className="flex-1 min-w-0">
                              <h3 className="text-2xl sm:text-3xl font-light text-foreground mb-2">
                                {selectedSkill.name}
                              </h3>
                              <div className="flex items-center gap-4">
                                <div className="flex items-center gap-2">
                                  <span className="text-sm text-muted-foreground font-mono">Proficiency</span>
                                  <span className="text-lg font-semibold text-purple-400">{selectedSkill.proficiency}%</span>
                                </div>
                              </div>
                            </div>
                            <button
                              onClick={() => setSelectedSkill(null)}
                              className="p-2 rounded-lg hover:bg-muted/30 transition-colors text-muted-foreground hover:text-foreground"
                              aria-label="Close"
                            >
                              <svg className="w-5 h-5" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" stroke="currentColor">
                                <path d="M6 18L18 6M6 6l12 12" />
                              </svg>
                            </button>
                          </div>

                          {/* Interactive Proficiency Bar */}
                          <div className="space-y-3">
                            <div className="flex justify-between items-center text-xs text-muted-foreground/60 font-mono">
                              <span>Beginner</span>
                              <span>Intermediate</span>
                              <span>Advanced</span>
                              <span>Expert</span>
                            </div>
                            <div className="relative h-3 bg-muted/30 rounded-full overflow-hidden border border-border/40">
                              <div
                                className="h-full bg-gradient-to-r from-purple-600 via-purple-500 to-purple-400 transition-all duration-1000 ease-out relative"
                                style={{ width: `${selectedSkill.proficiency}%` }}
                              >
                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse"></div>
                              </div>
                              {/* Markers */}
                              <div className="absolute inset-0 flex justify-between px-1">
                                {[25, 50, 75].map((mark) => (
                                  <div key={mark} className="w-px h-full bg-border/40"></div>
                                ))}
                              </div>
                            </div>
                          </div>

                          {/* Description */}
                          <div className="space-y-3">
                            <h4 className="text-sm font-semibold text-muted-foreground/80 uppercase tracking-wider">
                              About
                            </h4>
                            <p className="text-base text-muted-foreground leading-relaxed">
                              {selectedSkill.description}
                            </p>
                          </div>

                          {/* Usage Examples */}
                          <div className="space-y-3">
                            <h4 className="text-sm font-semibold text-muted-foreground/80 uppercase tracking-wider">
                              Used In
                            </h4>
                            <div className="p-4 rounded-lg bg-muted/20 border border-border/30">
                              <p className="text-sm text-muted-foreground leading-relaxed">
                                {selectedSkill.usedIn}
                              </p>
                            </div>
                          </div>

                          {/* Action Hint */}
                          <div className="pt-4 border-t border-border/30">
                            <p className="text-xs text-muted-foreground/60 italic text-center">
                              Click on other skills to view their details
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {skillGroup.skills.length > 4 && (
                      <div className="flex justify-center pt-4">
                        <button
                          onClick={() => toggleLoadMore(skillGroup.category)}
                          className="px-6 py-2 text-sm font-medium border border-border/60 rounded-lg hover:border-purple-500/40 hover:bg-gradient-to-r hover:from-purple-500/5 hover:to-transparent transition-all duration-300 text-muted-foreground hover:text-foreground"
                        >
                          {expandedSkills[skillGroup.category] ? "Show Less" : "Load More"}
                        </button>
                      </div>
                    )}
                  </div>
                ) : null,
              )}
            </div>
          </div>
        </section>

        {/* EXPERIENCE SECTION - Fourth */}
        <section
          id="work"
          ref={(el) => {
            if (el) sectionsRef.current[3] = el
          }}
          className="py-20 sm:py-32"
        >
          <div className="space-y-8 sm:space-y-12">
            <div className="flex items-end justify-between">
              <div className="space-y-2">
                <h2 className="text-3xl sm:text-4xl font-light section-header-glow">Experience</h2>
                <p className="text-sm text-muted-foreground/60 font-mono tracking-wider">2022 — Present</p>
              </div>
            </div>

            <div className="space-y-6">
              {experienceData.map((job, index) => (
                <button
                  key={index}
                  onClick={() => setSelectedExperience(job)}
                  className="w-full group grid grid-cols-12 gap-4 pb-6 border-b border-border/30 last:border-0 hover:border-purple-500/20 transition-all duration-300 text-left cursor-pointer"
                >
                  <div className="col-span-2 sm:col-span-1">
                    <div className="text-sm font-mono text-muted-foreground/60 group-hover:text-foreground/80 transition-colors">
                      {job.year}
                    </div>
                  </div>

                  <div className="col-span-10 sm:col-span-11 space-y-2">
                    <div>
                      <h3 className="text-base font-medium group-hover:text-purple-400 transition-colors">{job.role}</h3>
                      <div className="text-sm text-muted-foreground/70">{job.company}</div>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed">{job.description}</p>
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {job.tech.map((tech) => (
                        <span
                          key={tech}
                          className="px-2 py-0.5 text-xs text-muted-foreground/70 border border-border/40 rounded group-hover:border-purple-500/30 transition-colors"
                        >
                          {tech}
                        </span>
                      ))}
                    </div>
                    <div className="text-xs text-purple-400/70 group-hover:text-purple-400 pt-1">
                      Click to view details →
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* PROJECTS SECTION - Fifth */}
        <section
          id="projects"
          ref={(el) => {
            if (el) sectionsRef.current[4] = el
          }}
          className="py-20 sm:py-32"
        >
          <div className="space-y-8 sm:space-y-12">
            <div className="flex items-end justify-between">
              <div className="space-y-2">
                <h2 className="text-3xl sm:text-4xl font-light section-header-glow">Featured Projects</h2>
                <p className="text-sm text-muted-foreground/60 font-mono tracking-wider">
                  <Link
                    href="https://github.com/premalshah999"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-purple-400 transition-colors"
                  >
                    View all on GitHub →
                  </Link>
                </p>
              </div>
            </div>

            <div className="space-y-6">
              {projectsData.map((project, index) => (
                <button
                  key={index}
                  onClick={() => setSelectedProject(project)}
                  className="w-full group pb-6 border-b border-border/30 last:border-0 hover:border-purple-500/20 transition-all duration-300 text-left cursor-pointer"
                >
                  <div className="space-y-3">
                    <div>
                      <h3 className="text-base font-medium group-hover:text-purple-400 transition-colors">
                        {project.title}
                      </h3>
                      <p className="text-sm text-muted-foreground/80 mt-1">{project.description}</p>
                    </div>

                    <div className="flex flex-wrap gap-1.5">
                      {project.tech.map((tech) => (
                        <span
                          key={tech}
                          className="px-2 py-0.5 text-xs text-muted-foreground/70 border border-border/40 rounded group-hover:border-purple-500/30 transition-colors"
                        >
                          {tech}
                        </span>
                      ))}
                    </div>

                    <p className="text-xs text-muted-foreground/60 font-mono">{project.keyMetrics}</p>
                    <div className="text-xs text-purple-400/70 group-hover:text-purple-400">
                      Click to view details →
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </section>

        <section
          id="research"
          ref={(el) => {
            if (el) sectionsRef.current[5] = el
          }}
          className="py-20 sm:py-32"
        >
          <div className="space-y-8 sm:space-y-12">
            <div className="space-y-2">
              <h2 className="text-3xl sm:text-4xl font-light section-header-glow">Research & Publications</h2>
              <p className="text-sm text-muted-foreground/60 font-mono tracking-wider">Academic contributions & insights</p>
            </div>

            <div className="space-y-6">
              {researchData.map((publication, index) => (
                <button
                  key={index}
                  onClick={() => setSelectedResearch(publication)}
                  className="w-full block group pb-6 border-b border-border/30 last:border-0 hover:border-purple-500/20 transition-all duration-300 text-left cursor-pointer"
                >
                  <div className="space-y-2">
                    <div>
                      <h3 className="text-base font-medium group-hover:text-purple-400 transition-colors">
                        {publication.title}
                      </h3>
                      <p className="text-xs text-muted-foreground/60 font-mono mt-0.5">
                        {publication.publisher} • {publication.year}
                      </p>
                    </div>
                    <p className="text-sm text-muted-foreground/80">{publication.description}</p>
                    <div className="flex items-center gap-2 text-xs text-purple-400/70 group-hover:text-purple-400">
                      <span>Click to view details</span>
                      <svg
                        className="w-3 h-3 transform group-hover:translate-x-0.5 transition-transform"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M17 8l4 4m0 0l-4 4m4-4H3"
                        />
                      </svg>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* CONNECT SECTION - Last */}
        <section
          id="connect"
          ref={(el) => {
            if (el) sectionsRef.current[6] = el
          }}
          className="py-20 sm:py-32"
        >
          <div className="grid lg:grid-cols-2 gap-12 sm:gap-16">
            <div className="space-y-6 sm:space-y-8">
              <div className="space-y-2">
                <h2 className="text-3xl sm:text-4xl font-light section-header-glow">Let's Connect</h2>
                <p className="text-sm text-muted-foreground/60 font-mono tracking-wider">
                  Open to opportunities & collaborations
                </p>
              </div>

              <div className="space-y-6">
                <p className="text-lg sm:text-xl text-muted-foreground leading-relaxed">
                  Before you hire a consultant, reboot your laptop, or ask ChatGPT—Better Call Shah. I listen, I code,
                  and I reply (usually faster than your WiFi).
                </p>

                <div className="space-y-4">
                  <Link
                    href="mailto:premalshah204@gmail.com"
                    className="group flex items-center gap-3 text-foreground hover:text-purple-400 transition-colors duration-300"
                  >
                    <span className="text-base sm:text-lg">premalshah204@gmail.com</span>
                    <svg
                      className="w-5 h-5 transform group-hover:translate-x-1 transition-transform duration-300"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                    </svg>
                  </Link>
                </div>
              </div>
            </div>

            <div className="space-y-6 sm:space-y-8">
              <div className="text-sm text-muted-foreground font-mono">ELSEWHERE</div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {[
                  { name: "GitHub", handle: "@premalshah999", url: "https://github.com/premalshah999" },
                  { name: "LinkedIn", handle: "shahpremal", url: "https://www.linkedin.com/in/shahpremal/" },
                ].map((social) => (
                  <Link
                    key={social.name}
                    href={social.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group p-4 border border-border rounded-lg hover:border-purple-500/30 hover:bg-gradient-to-br hover:from-purple-500/5 hover:to-transparent transition-all duration-300 hover:shadow-sm"
                  >
                    <div className="space-y-2">
                      <div className="text-foreground group-hover:text-purple-400 transition-colors duration-300">
                        {social.name}
                      </div>
                      <div className="text-sm text-muted-foreground">{social.handle}</div>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </section>

        <footer className="py-12 sm:py-16 border-t border-border">
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6 sm:gap-8">
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">© 2025 Premal Shah. All rights reserved.</div>
            </div>
          </div>
        </footer>
      </main>

      {selectedSkill && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
          onClick={() => setSelectedSkill(null)}
        >
          <div
            className="bg-background border border-border rounded-lg p-6 sm:p-8 max-w-md w-full space-y-6 animate-fade-in-up shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="space-y-2 border-b border-border/20 pb-4">
              <div className="flex items-start justify-between gap-3">
                <h3 className="text-2xl sm:text-3xl font-medium pr-4">{selectedSkill.name}</h3>
                <button
                  onClick={() => setSelectedSkill(null)}
                  className="text-muted-foreground hover:text-foreground transition-colors p-1 flex-shrink-0"
                  aria-label="Close modal"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-muted-foreground/60 uppercase tracking-widest">
                  Proficiency Level
                </span>
                <span className="text-2xl font-bold text-foreground">{selectedSkill.proficiency}%</span>
              </div>
              <div className="h-2.5 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-muted-foreground/60 via-foreground/50 to-foreground/30 transition-all duration-700 rounded-full"
                  style={{ width: `${selectedSkill.proficiency}%` }}
                ></div>
              </div>
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Beginner</span>
                <span>Expert</span>
              </div>
            </div>

            <div className="space-y-3 pt-2">
              <div className="text-xs font-mono text-muted-foreground/60 uppercase tracking-widest">What is it?</div>
              <p className="text-muted-foreground leading-relaxed text-sm">{selectedSkill.description}</p>
            </div>

            <div className="space-y-3 pt-2">
              <div className="text-xs font-mono text-muted-foreground/60 uppercase tracking-widest">
                Where I've Used It
              </div>
              <p className="text-muted-foreground leading-relaxed text-sm">{selectedSkill.usedIn}</p>
            </div>

            <div className="flex gap-2 pt-4">
              <button
                onClick={() => setSelectedSkill(null)}
                className="flex-1 px-4 py-2.5 border border-border rounded hover:border-muted-foreground/50 hover:bg-muted/30 transition-all duration-300 text-sm font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Experience Modal */}
      {selectedExperience && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in"
          onClick={() => setSelectedExperience(null)}
        >
          <div
            className="bg-background border border-purple-500/30 rounded-xl p-6 sm:p-8 max-w-2xl w-full max-h-[90vh] overflow-y-auto space-y-6 animate-fade-in-up shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-start justify-between gap-4 border-b border-border/30 pb-4">
              <div className="space-y-2 flex-1">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  <span className="text-xs font-mono text-muted-foreground/60">{selectedExperience.duration}</span>
                </div>
                <h3 className="text-2xl sm:text-3xl font-light text-foreground">{selectedExperience.role}</h3>
                <div className="space-y-0.5">
                  <p className="text-base text-purple-400">{selectedExperience.company}</p>
                  <p className="text-sm text-muted-foreground/60">{selectedExperience.location}</p>
                </div>
              </div>
              <button
                onClick={() => setSelectedExperience(null)}
                className="p-2 rounded-lg hover:bg-muted/30 transition-colors text-muted-foreground hover:text-foreground flex-shrink-0"
                aria-label="Close"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Key Achievements */}
            <div className="space-y-3">
              <h4 className="text-sm font-mono text-muted-foreground/60 uppercase tracking-wider">Key Achievements</h4>
              <div className="space-y-3">
                {selectedExperience.achievements.map((achievement: string, idx: number) => (
                  <div key={idx} className="flex gap-3 items-start">
                    <div className="w-1.5 h-1.5 bg-purple-500/60 rounded-full mt-2 flex-shrink-0"></div>
                    <p className="text-sm text-muted-foreground leading-relaxed">{achievement}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Technologies */}
            <div className="space-y-3 pt-2">
              <h4 className="text-sm font-mono text-muted-foreground/60 uppercase tracking-wider">Technologies</h4>
              <div className="flex flex-wrap gap-2">
                {selectedExperience.tech.map((tech: string) => (
                  <span
                    key={tech}
                    className="px-3 py-1.5 text-xs text-foreground/80 bg-muted/30 border border-border/40 rounded-full hover:border-purple-500/40 transition-colors"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Project Modal */}
      {selectedProject && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in"
          onClick={() => setSelectedProject(null)}
        >
          <div
            className="bg-background border border-purple-500/30 rounded-xl p-6 sm:p-8 max-w-4xl w-full max-h-[90vh] overflow-y-auto space-y-6 animate-fade-in-up shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-start justify-between gap-4 border-b border-border/30 pb-4">
              <div className="space-y-2 flex-1">
                <h3 className="text-2xl sm:text-3xl font-light text-foreground">{selectedProject.title}</h3>
                <p className="text-sm text-muted-foreground/80 leading-relaxed">{selectedProject.description}</p>
              </div>
              <button
                onClick={() => setSelectedProject(null)}
                className="p-2 rounded-lg hover:bg-muted/30 transition-colors text-muted-foreground hover:text-foreground flex-shrink-0"
                aria-label="Close"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Video Demo Player */}
            {selectedProject.videoUrl && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-mono text-muted-foreground/60 uppercase tracking-wider">Demo Video</h4>
                  <div className="flex items-center gap-2 text-xs text-purple-400/70">
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                    <span>Live Preview</span>
                  </div>
                </div>
                <div className="relative aspect-video bg-black rounded-lg overflow-hidden border border-purple-500/20">
                  <video
                    src={selectedProject.videoUrl}
                    controls
                    className="w-full h-full"
                    playsInline
                    preload="metadata"
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>
              </div>
            )}

            {/* Metrics */}
            <div className="p-4 rounded-lg bg-purple-500/5 border border-purple-500/20">
              <p className="text-sm font-mono text-purple-400/80">{selectedProject.keyMetrics}</p>
            </div>

            {/* Achievements */}
            <div className="space-y-3">
              <h4 className="text-sm font-mono text-muted-foreground/60 uppercase tracking-wider">Key Features</h4>
              <div className="space-y-3">
                {selectedProject.achievements.map((achievement: string, idx: number) => (
                  <div key={idx} className="flex gap-3 items-start">
                    <div className="w-1.5 h-1.5 bg-purple-500/60 rounded-full mt-2 flex-shrink-0"></div>
                    <p className="text-sm text-muted-foreground leading-relaxed">{achievement}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Technologies */}
            <div className="space-y-3">
              <h4 className="text-sm font-mono text-muted-foreground/60 uppercase tracking-wider">Tech Stack</h4>
              <div className="flex flex-wrap gap-2">
                {selectedProject.tech.map((tech: string) => (
                  <span
                    key={tech}
                    className="px-3 py-1.5 text-xs text-foreground/80 bg-muted/30 border border-border/40 rounded-full hover:border-purple-500/40 transition-colors"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3 pt-4 border-t border-border/30">
              <Link
                href={selectedProject.demoUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-1 px-4 py-2.5 bg-purple-500/10 border border-purple-500/30 rounded-lg hover:bg-purple-500/20 transition-all duration-300 text-sm font-medium text-center text-purple-400 hover:text-purple-300"
              >
                View Demo
              </Link>
              <Link
                href={selectedProject.githubUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-1 px-4 py-2.5 border border-border rounded-lg hover:border-purple-500/40 hover:bg-muted/30 transition-all duration-300 text-sm font-medium text-center"
              >
                GitHub
              </Link>
            </div>
          </div>
        </div>
      )}

      {/* Research Modal */}
      {selectedResearch && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in"
          onClick={() => setSelectedResearch(null)}
        >
          <div
            className="bg-background border border-purple-500/30 rounded-xl p-6 sm:p-8 max-w-2xl w-full max-h-[90vh] overflow-y-auto space-y-6 animate-fade-in-up shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-start justify-between gap-4 border-b border-border/30 pb-4">
              <div className="space-y-2 flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs font-mono text-purple-400/80 px-2 py-0.5 bg-purple-500/10 rounded">
                    {selectedResearch.year}
                  </span>
                  <span className="text-xs font-mono text-muted-foreground/60">{selectedResearch.publisher}</span>
                </div>
                <h3 className="text-xl sm:text-2xl font-light text-foreground leading-snug">
                  {selectedResearch.title}
                </h3>
              </div>
              <button
                onClick={() => setSelectedResearch(null)}
                className="p-2 rounded-lg hover:bg-muted/30 transition-colors text-muted-foreground hover:text-foreground flex-shrink-0"
                aria-label="Close"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Abstract */}
            <div className="space-y-3">
              <h4 className="text-sm font-mono text-muted-foreground/60 uppercase tracking-wider">Abstract</h4>
              <p className="text-sm text-muted-foreground leading-relaxed">{selectedResearch.abstract}</p>
            </div>

            {/* Keywords */}
            <div className="space-y-3">
              <h4 className="text-sm font-mono text-muted-foreground/60 uppercase tracking-wider">Keywords</h4>
              <div className="flex flex-wrap gap-2">
                {selectedResearch.keywords.map((keyword: string) => (
                  <span
                    key={keyword}
                    className="px-3 py-1.5 text-xs text-foreground/80 bg-muted/30 border border-border/40 rounded-full"
                  >
                    {keyword}
                  </span>
                ))}
              </div>
            </div>

            {/* Action Button */}
            <div className="pt-4 border-t border-border/30">
              <Link
                href={selectedResearch.url}
                target="_blank"
                rel="noopener noreferrer"
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-purple-500/10 border border-purple-500/30 rounded-lg hover:bg-purple-500/20 transition-all duration-300 text-sm font-medium text-purple-400 hover:text-purple-300"
              >
                <span>Read Full Paper</span>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </Link>
            </div>
          </div>
        </div>
      )}

      <div className="fixed bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-background via-background/80 to-transparent pointer-events-none"></div>
    </div>
  )
}
