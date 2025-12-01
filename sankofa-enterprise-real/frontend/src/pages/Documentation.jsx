import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  BookOpen, Brain, AlertTriangle, Package, Layers, BarChart3, Code,
  ChevronLeft, ChevronRight, Loader2, BookMarked, Microscope, Database, MessageSquare, Share2
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card.jsx';

const docs = [
  { 
    id: 'sankofa', 
    label: 'Sankofa 101', 
    icon: BookOpen, 
    file: 'USE_A_CABECA_SANKOFA.md',
    description: 'Introdução ao sistema de detecção de fraudes'
  },
  { 
    id: 'ml', 
    label: 'Machine Learning', 
    icon: Brain, 
    file: 'USE_A_CABECA_ML.md',
    description: 'Como funciona a inteligência artificial'
  },
  { 
    id: 'fraudes', 
    label: 'Tipos de Fraudes', 
    icon: AlertTriangle, 
    file: 'USE_A_CABECA_FRAUDES.md',
    description: 'Enciclopédia completa de fraudes bancárias'
  },
  { 
    id: 'research-ml', 
    label: 'Módulos de Pesquisa ML', 
    icon: Microscope, 
    file: 'MODULOS_PESQUISA_ML.md',
    description: 'Bahnsen, PIX Taxonomy, NLP e Transfer Learning'
  },
  { 
    id: 'datasets', 
    label: 'Datasets', 
    icon: Database, 
    file: 'DataSets.md',
    description: 'Catálogo de datasets para treinamento'
  },
  { 
    id: 'payload', 
    label: 'Payload de Entrada', 
    icon: Package, 
    file: 'PAYLOAD_ENTRADA.md',
    description: 'Estrutura dos dados de transação'
  },
  { 
    id: 'funcional', 
    label: 'Doc. Funcional', 
    icon: Layers, 
    file: 'DOCUMENTACAO_FUNCIONAL.md',
    description: 'Especificação funcional do sistema'
  },
  { 
    id: 'diagramas', 
    label: 'Diagramas', 
    icon: BarChart3, 
    file: 'DIAGRAMAS.md',
    description: 'Fluxogramas e diagramas técnicos'
  },
  { 
    id: 'arquitetura', 
    label: 'Arquitetura', 
    icon: Code, 
    file: 'ARQUITETURA_TECNICA.md',
    description: 'Arquitetura técnica completa'
  }
];

export function Documentation() {
  const [activeDoc, setActiveDoc] = useState(0);
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadDocument = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`/docs/${docs[activeDoc].file}`);
        if (!response.ok) {
          throw new Error('Documento não encontrado');
        }
        let text = await response.text();
        text = text.replace(/!\[([^\]]*)\]\(images\/([^)]+)\)/g, '![$1](/docs/images/$2)');
        setContent(text);
      } catch (err) {
        setError(err.message);
        setContent('');
      } finally {
        setLoading(false);
      }
    };
    loadDocument();
  }, [activeDoc]);

  const goToPrev = () => {
    if (activeDoc > 0) setActiveDoc(activeDoc - 1);
  };

  const goToNext = () => {
    if (activeDoc < docs.length - 1) setActiveDoc(activeDoc + 1);
  };

  const CurrentIcon = docs[activeDoc].icon;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="flex flex-col lg:flex-row">
        <aside className="w-full lg:w-72 bg-white dark:bg-gray-800 border-b lg:border-b-0 lg:border-r border-gray-200 dark:border-gray-700 lg:min-h-[calc(100vh-4rem)]">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3">
              <BookMarked className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">Documentação</h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">Sankofa Enterprise Pro</p>
              </div>
            </div>
          </div>
          <nav className="p-2 space-y-1 overflow-y-auto max-h-[300px] lg:max-h-none">
            {docs.map((doc, index) => {
              const Icon = doc.icon;
              return (
                <button
                  key={doc.id}
                  onClick={() => setActiveDoc(index)}
                  className={`w-full flex items-start gap-3 p-3 rounded-lg text-left transition ${
                    activeDoc === index
                      ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-l-4 border-blue-600'
                      : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  <Icon className={`h-5 w-5 mt-0.5 flex-shrink-0 ${
                    activeDoc === index ? 'text-blue-600' : 'text-gray-400'
                  }`} />
                  <div>
                    <p className="font-medium text-sm">{doc.label}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{doc.description}</p>
                  </div>
                </button>
              );
            })}
          </nav>
          <div className="p-4 border-t border-gray-200 dark:border-gray-700 mt-auto">
            <div className="text-xs text-gray-500 dark:text-gray-400">
              <p className="font-semibold text-gray-700 dark:text-gray-300">9 Documentos</p>
              <p>12.000+ linhas de conteúdo</p>
              <p>Metodologia Head First</p>
              <p className="mt-1 text-green-600 dark:text-green-400">+ Módulos ML v2.0</p>
            </div>
          </div>
        </aside>

        <main className="flex-1 p-4 lg:p-8 overflow-y-auto">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                  <CurrentIcon className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                    {docs[activeDoc].label}
                  </h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {docs[activeDoc].description}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={goToPrev}
                  disabled={activeDoc === 0}
                  className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
                >
                  <ChevronLeft className="h-5 w-5" />
                </button>
                <span className="text-sm text-gray-500 dark:text-gray-400 px-2">
                  {activeDoc + 1} / {docs.length}
                </span>
                <button
                  onClick={goToNext}
                  disabled={activeDoc === docs.length - 1}
                  className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
                >
                  <ChevronRight className="h-5 w-5" />
                </button>
              </div>
            </div>

            <Card className="shadow-lg">
              <CardContent className="p-6 lg:p-8">
                {loading ? (
                  <div className="flex flex-col items-center justify-center py-20">
                    <Loader2 className="h-10 w-10 text-blue-600 animate-spin mb-4" />
                    <p className="text-gray-500 dark:text-gray-400">Carregando documento...</p>
                  </div>
                ) : error ? (
                  <div className="text-center py-20">
                    <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
                    <p className="text-red-600 dark:text-red-400 font-semibold">Erro ao carregar</p>
                    <p className="text-gray-500 dark:text-gray-400 text-sm mt-2">{error}</p>
                  </div>
                ) : (
                  <article className="prose prose-lg dark:prose-invert max-w-none
                    prose-headings:font-bold prose-headings:text-gray-900 dark:prose-headings:text-white
                    prose-h1:text-3xl prose-h1:border-b prose-h1:border-gray-200 dark:prose-h1:border-gray-700 prose-h1:pb-4 prose-h1:mb-6
                    prose-h2:text-2xl prose-h2:mt-8 prose-h2:mb-4
                    prose-h3:text-xl prose-h3:mt-6 prose-h3:mb-3
                    prose-p:text-gray-700 dark:prose-p:text-gray-300 prose-p:leading-relaxed
                    prose-a:text-blue-600 dark:prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline
                    prose-strong:text-gray-900 dark:prose-strong:text-white
                    prose-code:bg-gray-100 dark:prose-code:bg-gray-800 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none
                    prose-pre:bg-gray-900 dark:prose-pre:bg-gray-950 prose-pre:text-gray-100 prose-pre:rounded-lg prose-pre:shadow-lg prose-pre:overflow-x-auto
                    prose-table:border-collapse prose-table:w-full
                    prose-th:bg-gray-100 dark:prose-th:bg-gray-800 prose-th:p-3 prose-th:text-left prose-th:border prose-th:border-gray-300 dark:prose-th:border-gray-600
                    prose-td:p-3 prose-td:border prose-td:border-gray-300 dark:prose-td:border-gray-600
                    prose-tr:even:bg-gray-50 dark:prose-tr:even:bg-gray-800/50
                    prose-ul:list-disc prose-ul:pl-6
                    prose-ol:list-decimal prose-ol:pl-6
                    prose-li:text-gray-700 dark:prose-li:text-gray-300 prose-li:my-1
                    prose-blockquote:border-l-4 prose-blockquote:border-blue-500 prose-blockquote:bg-blue-50 dark:prose-blockquote:bg-blue-900/20 prose-blockquote:p-4 prose-blockquote:rounded-r-lg prose-blockquote:not-italic
                    prose-hr:border-gray-200 dark:prose-hr:border-gray-700 prose-hr:my-8
                    prose-img:rounded-lg prose-img:shadow-md
                  ">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      components={{
                        img: ({node, src, alt, ...props}) => (
                          <img 
                            src={src} 
                            alt={alt || 'Imagem da documentação'} 
                            className="rounded-lg shadow-lg max-w-full h-auto my-6 mx-auto block border border-gray-200 dark:border-gray-700"
                            loading="lazy"
                            {...props} 
                          />
                        )
                      }}
                    >
                      {content}
                    </ReactMarkdown>
                  </article>
                )}
              </CardContent>
            </Card>

            <div className="flex justify-between items-center mt-6">
              <button
                onClick={goToPrev}
                disabled={activeDoc === 0}
                className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                <ChevronLeft className="h-4 w-4" />
                <span className="text-sm font-medium">
                  {activeDoc > 0 ? docs[activeDoc - 1].label : 'Anterior'}
                </span>
              </button>
              <button
                onClick={goToNext}
                disabled={activeDoc === docs.length - 1}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                <span className="text-sm font-medium">
                  {activeDoc < docs.length - 1 ? docs[activeDoc + 1].label : 'Próximo'}
                </span>
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
