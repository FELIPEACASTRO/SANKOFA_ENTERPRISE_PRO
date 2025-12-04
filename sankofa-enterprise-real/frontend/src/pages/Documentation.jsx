import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  BookOpen, Brain, AlertTriangle, Package, Layers, BarChart3, Code,
  ChevronLeft, ChevronRight, Loader2, BookMarked, Microscope, Database, 
  MessageSquare, Share2, Shield, FileCheck, TestTube, CheckCircle,
  Cpu, Target, Award, Sparkles, GraduationCap, ClipboardList, Zap
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card.jsx';

const docs = [
  { 
    id: 'sankofa', 
    label: 'Sankofa 101', 
    icon: BookOpen, 
    file: 'USE_A_CABECA_SANKOFA.md',
    description: 'Introdução completa ao sistema de detecção de fraudes',
    category: 'intro',
    badge: 'Essencial'
  },
  { 
    id: 'ml', 
    label: 'Machine Learning', 
    icon: Brain, 
    file: 'USE_A_CABECA_ML.md',
    description: 'Como funciona a inteligência artificial do sistema',
    category: 'ml',
    badge: null
  },
  { 
    id: 'ml-completo', 
    label: 'Guia Completo ML', 
    icon: Brain, 
    file: 'GUIA_COMPLETO_ML.md',
    description: 'Arquitetura: 7 módulos, ensemble stacking, fórmulas matemáticas',
    category: 'ml',
    badge: 'Avançado'
  },
  { 
    id: 'fraudes', 
    label: 'Tipos de Fraudes', 
    icon: AlertTriangle, 
    file: 'USE_A_CABECA_FRAUDES.md',
    description: 'Enciclopédia completa de fraudes bancárias PIX/Cartão',
    category: 'fraudes',
    badge: null
  },
  { 
    id: 'research-ml', 
    label: 'Módulos de Pesquisa', 
    icon: Microscope, 
    file: 'MODULOS_PESQUISA_ML.md',
    description: 'Bahnsen Features, PIX Taxonomy, NLP, Transfer Learning',
    category: 'ml',
    badge: 'Pesquisa'
  },
  { 
    id: 'datasets-features', 
    label: 'Datasets e Features', 
    icon: Database, 
    file: 'DATASETS_FEATURES_MODELOS.md',
    description: '7 datasets acadêmicos, 100+ features, 13 modelos de ML',
    category: 'dados',
    badge: null
  },
  { 
    id: 'datasets', 
    label: 'Histórias de Fraude', 
    icon: Database, 
    file: 'DataSets.md',
    description: '50 casos reais de fraude baseados em dados reais',
    category: 'fraudes',
    badge: null
  },
  { 
    id: 'hard-rules', 
    label: '216 Regras Duras', 
    icon: Shield, 
    file: 'HARD_RULES_216.md',
    description: 'Engine unificado: BACEN, PIX, Malware, Velocity, Social Engineering',
    category: 'regras',
    badge: 'v2.0'
  },
  { 
    id: 'payload', 
    label: 'Payload de Entrada', 
    icon: Package, 
    file: 'PAYLOAD_ENTRADA.md',
    description: 'Estrutura completa dos dados de transação enviados à API',
    category: 'tecnico',
    badge: null
  },
  { 
    id: 'funcional', 
    label: 'Doc. Funcional', 
    icon: Layers, 
    file: 'DOCUMENTACAO_FUNCIONAL.md',
    description: 'Especificação funcional completa do sistema',
    category: 'tecnico',
    badge: null
  },
  { 
    id: 'diagramas', 
    label: 'Diagramas', 
    icon: BarChart3, 
    file: 'DIAGRAMAS.md',
    description: 'Fluxogramas, diagramas de arquitetura e sequência',
    category: 'tecnico',
    badge: null
  },
  { 
    id: 'arquitetura', 
    label: 'Arquitetura Técnica', 
    icon: Code, 
    file: 'ARQUITETURA_TECNICA.md',
    description: 'Arquitetura técnica completa: backend, frontend, ML, infra',
    category: 'tecnico',
    badge: null
  },
  { 
    id: 'db-postgres', 
    label: 'PostgreSQL', 
    icon: Database, 
    file: 'DB_01_POSTGRES_INVENTARIO_ULTRA_MILITAR.md',
    description: '16 tabelas, índices otimizados, queries de alta performance',
    category: 'banco',
    badge: null
  },
  { 
    id: 'db-redis', 
    label: 'Redis Cache', 
    icon: Database, 
    file: 'DB_03_REDIS_ANALISE_MILITAR.md',
    description: 'Cache distribuído, TTL dinâmico, fallback automático',
    category: 'banco',
    badge: null
  },
  { 
    id: 'qa-relatorio', 
    label: 'Relatório QA', 
    icon: TestTube, 
    file: 'RELATORIO_QA.md',
    description: 'Relatório completo de Quality Assurance e testes',
    category: 'qa',
    badge: 'Novo'
  },
  { 
    id: 'triple-check', 
    label: 'Triple Check', 
    icon: CheckCircle, 
    file: 'TRIPLE_CHECK_AUDITORIA.md',
    description: 'Auditoria tripla: código, segurança, performance',
    category: 'qa',
    badge: null
  },
  { 
    id: 'blueprint', 
    label: 'Blueprint 300M', 
    icon: Zap, 
    file: 'BLUEPRINT_MOTOR_FRAUDE_300M.md',
    description: 'Motor de fraude para 300M requisições/dia',
    category: 'tecnico',
    badge: 'Enterprise'
  },
  { 
    id: 'manual-usuario', 
    label: 'Manual do Usuário', 
    icon: GraduationCap, 
    file: 'MANUAL_USUARIO.md',
    description: 'Guia prático para analistas de fraude',
    category: 'intro',
    badge: 'Essencial'
  }
];

const categoryColors = {
  intro: 'bg-blue-500',
  ml: 'bg-purple-500',
  fraudes: 'bg-red-500',
  regras: 'bg-orange-500',
  tecnico: 'bg-gray-500',
  dados: 'bg-green-500',
  banco: 'bg-cyan-500',
  qa: 'bg-yellow-500'
};

const categoryLabels = {
  intro: 'Introdução',
  ml: 'Machine Learning',
  fraudes: 'Fraudes',
  regras: 'Regras',
  tecnico: 'Técnico',
  dados: 'Dados',
  banco: 'Banco de Dados',
  qa: 'Quality Assurance'
};

export function Documentation() {
  const [activeDoc, setActiveDoc] = useState(0);
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

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

  const filteredDocs = docs.filter(doc => {
    const matchesSearch = doc.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          doc.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || doc.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const goToPrev = () => {
    if (activeDoc > 0) setActiveDoc(activeDoc - 1);
  };

  const goToNext = () => {
    if (activeDoc < docs.length - 1) setActiveDoc(activeDoc + 1);
  };

  const CurrentIcon = docs[activeDoc].icon;
  const categories = [...new Set(docs.map(d => d.category))];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="flex flex-col lg:flex-row">
        <aside className="w-full lg:w-80 bg-white dark:bg-gray-800 border-b lg:border-b-0 lg:border-r border-gray-200 dark:border-gray-700 lg:min-h-[calc(100vh-4rem)]">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4">
              <BookMarked className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">Documentação</h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">Sankofa Enterprise Pro v2.1</p>
              </div>
            </div>
            <input
              type="text"
              placeholder="Buscar documentos..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="flex flex-wrap gap-1 mt-3">
              <button
                onClick={() => setSelectedCategory('all')}
                className={`px-2 py-1 text-xs rounded-full ${selectedCategory === 'all' ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'}`}
              >
                Todos
              </button>
              {categories.map(cat => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-2 py-1 text-xs rounded-full ${selectedCategory === cat ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'}`}
                >
                  {categoryLabels[cat]}
                </button>
              ))}
            </div>
          </div>
          
          <nav className="p-2 space-y-1 overflow-y-auto max-h-[400px] lg:max-h-[calc(100vh-280px)]">
            {filteredDocs.map((doc) => {
              const docIndex = docs.findIndex(d => d.id === doc.id);
              const Icon = doc.icon;
              return (
                <button
                  key={doc.id}
                  onClick={() => setActiveDoc(docIndex)}
                  className={`w-full flex items-start gap-3 p-3 rounded-lg text-left transition ${
                    activeDoc === docIndex
                      ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-l-4 border-blue-600'
                      : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  <Icon className={`h-5 w-5 mt-0.5 flex-shrink-0 ${
                    activeDoc === docIndex ? 'text-blue-600' : 'text-gray-400'
                  }`} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="font-medium text-sm truncate">{doc.label}</p>
                      {doc.badge && (
                        <span className={`px-1.5 py-0.5 text-[10px] font-semibold rounded ${
                          doc.badge === 'Essencial' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                          doc.badge === 'Novo' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                          doc.badge === 'Avançado' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400' :
                          doc.badge === 'Pesquisa' ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400' :
                          'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                        }`}>
                          {doc.badge}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 line-clamp-2">{doc.description}</p>
                  </div>
                </button>
              );
            })}
          </nav>
          
          <div className="p-4 border-t border-gray-200 dark:border-gray-700 mt-auto">
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <Award className="h-5 w-5 text-blue-600" />
                <span className="font-bold text-sm text-gray-900 dark:text-white">Status: 10/10</span>
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                <p className="flex items-center gap-1">
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  18 Documentos disponíveis
                </p>
                <p className="flex items-center gap-1">
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  30.000+ linhas de conteúdo
                </p>
                <p className="flex items-center gap-1">
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  Metodologia Head First
                </p>
                <p className="flex items-center gap-1">
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  1.397+ testes validados
                </p>
              </div>
            </div>
          </div>
        </aside>

        <main className="flex-1 p-4 lg:p-8 overflow-y-auto">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${categoryColors[docs[activeDoc].category]} bg-opacity-20`}>
                  <CurrentIcon className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                      {docs[activeDoc].label}
                    </h2>
                    <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${categoryColors[docs[activeDoc].category]} text-white`}>
                      {categoryLabels[docs[activeDoc].category]}
                    </span>
                  </div>
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
