"""Generate Markdown summary from JSON audit report"""
import json

# Ler o relatório JSON
with open('ml_audit_detailed.json', 'r', encoding='utf-8') as f:
    audit = json.load(f)

# Criar relatório Markdown
md_content = []
md_content.append('# Relatório de Auditoria - ML Engine')
md_content.append('')
md_content.append(f"**Data:** {audit['metadata']['audit_date']}")
md_content.append(f"**Diretório:** `{audit['metadata']['directory']}`")
md_content.append('')

# Sumário
md_content.append('## Sumário Executivo')
md_content.append('')
md_content.append(f"- **Total de arquivos analisados:** {audit['metadata']['total_files_analyzed']}")
md_content.append(f"- **Total de classes:** {audit['summary']['total_classes']}")
md_content.append(f"- **Total de métodos:** {audit['summary']['total_methods']}")
md_content.append(f"- **Métodos incompletos:** {audit['summary']['incomplete_methods']}")
md_content.append(f"- **Arquivos com problemas:** {audit['summary']['files_with_issues']}")
md_content.append('')

# Imports opcionais
md_content.append('## Bibliotecas Opcionais Detectadas')
md_content.append('')
md_content.append(f"Total de imports opcionais únicos: {len(audit['summary']['optional_imports_unique'])}")
md_content.append('')
for imp in sorted(audit['summary']['optional_imports_unique']):
    md_content.append(f"- `{imp}`")
md_content.append('')

# Análise por arquivo (top problemas)
md_content.append('## Arquivos com Problemas')
md_content.append('')

problem_files = []
for file_name, file_data in audit['files'].items():
    if file_data.get('problemas'):
        problem_files.append((file_name, file_data))

if problem_files:
    for file_name, file_data in sorted(problem_files):
        md_content.append(f"### {file_name}")
        md_content.append('')
        md_content.append(f"- **Pode importar:** {file_data['pode_importar']}")
        if file_data['erro_import']:
            md_content.append(f"- **Erro:** `{file_data['erro_import']}`")
        md_content.append('- **Problemas:**')
        for problema in file_data['problemas']:
            md_content.append(f"  - {problema}")
        md_content.append('')
else:
    md_content.append('Nenhum problema crítico detectado.')
    md_content.append('')

# Métodos incompletos
md_content.append('## Métodos Incompletos')
md_content.append('')

incomplete_found = False
for file_name, file_data in sorted(audit['files'].items()):
    for class_name, class_info in file_data['classes'].items():
        if class_info['metodos_incompletos']:
            incomplete_found = True
            md_content.append(f"### {file_name} - {class_name}")
            md_content.append('')
            for metodo in class_info['metodos_incompletos']:
                md_content.append(f"- `{metodo}` (stub/NotImplementedError)")
            md_content.append('')

if not incomplete_found:
    md_content.append('Nenhum método incompleto detectado.')
    md_content.append('')

# Estatísticas de Classes ML
md_content.append('## Classes com Funcionalidade ML')
md_content.append('')

ml_classes = []
for file_name, file_data in audit['files'].items():
    for class_name, class_info in file_data['classes'].items():
        has_ml = any([
            class_info.get('tem_train'),
            class_info.get('tem_predict'),
            class_info.get('tem_fit'),
            class_info.get('tem_analyze'),
            class_info.get('tem_detect')
        ])
        if has_ml:
            ml_classes.append({
                'file': file_name,
                'class': class_name,
                'train': class_info.get('tem_train', False),
                'predict': class_info.get('tem_predict', False),
                'fit': class_info.get('tem_fit', False),
                'analyze': class_info.get('tem_analyze', False),
                'detect': class_info.get('tem_detect', False)
            })

md_content.append(f"Total de classes ML: {len(ml_classes)}")
md_content.append('')
md_content.append('| Arquivo | Classe | train | predict | fit | analyze | detect |')
md_content.append('|---------|--------|-------|---------|-----|---------|--------|')

for ml_class in sorted(ml_classes, key=lambda x: (x['file'], x['class'])):
    train_icon = '✓' if ml_class['train'] else '✗'
    predict_icon = '✓' if ml_class['predict'] else '✗'
    fit_icon = '✓' if ml_class['fit'] else '✗'
    analyze_icon = '✓' if ml_class['analyze'] else '✗'
    detect_icon = '✓' if ml_class['detect'] else '✗'

    md_content.append(f"| {ml_class['file']} | {ml_class['class']} | {train_icon} | {predict_icon} | {fit_icon} | {analyze_icon} | {detect_icon} |")

md_content.append('')

# Salvar relatório Markdown
with open('ml_audit_summary.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_content))

print('Relatório Markdown criado: ml_audit_summary.md')
print(f'Total de classes ML: {len(ml_classes)}')
