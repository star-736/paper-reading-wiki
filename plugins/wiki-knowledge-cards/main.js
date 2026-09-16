const { Plugin, ItemView, Component, MarkdownRenderer, Notice } = require('obsidian');
const VIEW = 'wiki-knowledge-card';
const TYPES = { concepts: '概念', models: '模型', comparisons: '比较' };
const EXCLUDED = new Set(['来源', '范围', '相关页面', '待追问', '后续问题']);

function category(path) {
  const match = /^wiki\/(concepts|models|comparisons)\/[^/]+\.md$/.exec(path);
  return match ? match[1] : null;
}

function extractCard(text, path, frontmatter = {}) {
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/);
  let start = 0;
  if (lines[0] === '---') {
    const end = lines.findIndex((line, i) => i > 0 && /^(---|\.\.\.)\s*$/.test(line));
    if (end > 0) start = end + 1;
  }
  const headings = [];
  let title = '', fence = null;
  for (let i = start; i < lines.length; i++) {
    const f = /^ {0,3}(`{3,}|~{3,})/.exec(lines[i]);
    if (f) {
      if (!fence) fence = f[1];
      else if (f[1][0] === fence[0] && f[1].length >= fence.length) fence = null;
      continue;
    }
    if (fence) continue;
    const h = /^ {0,3}(#{1,2})\s+(.+?)\s*#*\s*$/.exec(lines[i]);
    if (!h) continue;
    if (h[1].length === 1 && !title) title = h[2];
    headings.push({ level: h[1].length, heading: h[2], line: i });
  }
  const sections = headings.filter(h => h.level === 2).map(h => {
    const next = headings.find(n => n.line > h.line);
    return { ...h, content: lines.slice(h.line + 1, next ? next.line : lines.length).join('\n').trim() };
  });
  const kind = category(path);
  const priorities = kind === 'concepts' ? ['定义'] : kind === 'models' ? ['身份'] : ['速览', '主综合', '几个值得记住的判断'];
  const selected = priorities.map(name => sections.find(s => s.heading === name && s.content)).find(Boolean)
    || sections.find(s => s.content && !EXCLUDED.has(s.heading));
  let body = selected ? selected.content : sections.length ? '' : lines.slice(start).join('\n').replace(/^\s*# .+\n?/, '').trim();
  return {
    path, kind, title: typeof frontmatter.title === 'string' && frontmatter.title.trim() ? frontmatter.title : title || path.split('/').pop().replace(/\.md$/, ''),
    description: typeof frontmatter.description === 'string' ? frontmatter.description : '',
    heading: selected?.heading || '', line: selected?.line || 0,
    body: body || '此页暂无可展示的核心小节，请打开原文阅读。'
  };
}

function draw(paths, state, random = Math.random) {
  const all = [...new Set(paths)].sort();
  if (!all.length) return { path: null, state: { seen: [], last: state.last || null } };
  let seen = (Array.isArray(state.seen) ? state.seen : []).filter(p => all.includes(p));
  let pool = all.filter(p => !seen.includes(p));
  if (!pool.length) { seen = []; pool = all.filter(p => all.length === 1 || p !== state.last); }
  const path = pool[Math.floor(random() * pool.length)];
  return { path, state: { seen: [...seen, path], last: path } };
}

class CardView extends ItemView {
  constructor(leaf, plugin) { super(leaf); this.plugin = plugin; this.request = 0; }
  getViewType() { return VIEW; }
  getDisplayText() { return '知识卡片'; }
  getIcon() { return 'layers'; }
  async onOpen() {
    this.contentEl.empty();
    this.contentEl.addClass('wkc-view');
    this.shell = this.contentEl.createDiv({ cls: 'wkc-shell' });
    const masthead = this.shell.createDiv({ cls: 'wkc-masthead' });
    masthead.createSpan({ text: '知识卡片', cls: 'wkc-brand' });
    masthead.createSpan({ text: '从一次偶遇，开始理解', cls: 'wkc-tagline' });
    this.card = this.shell.createEl('article', { cls: 'wkc-card' });
    this.reading = this.card.createDiv({ cls: 'wkc-reading' });
    this.reading.setAttribute('aria-live', 'polite');
    this.reading.setAttribute('tabindex', '0');
    this.footer = this.card.createDiv({ cls: 'wkc-footer' });
    this.nextButton = this.footer.createEl('button', { text: '换一张' });
    this.openButton = this.footer.createEl('button', { text: '打开原文', cls: 'mod-cta' });
    this.openButton.disabled = true;
    this.registerDomEvent(this.nextButton, 'click', () => this.next());
    this.registerDomEvent(this.openButton, 'click', () => this.openOriginal());
    this.registerDomEvent(this.reading, 'click', event => {
      const link = event.target.closest?.('a.internal-link');
      if (!link || !this.current) return;
      const target = link.getAttribute('data-href') || link.getAttribute('href');
      if (!target) return;
      event.preventDefault();
      event.stopPropagation();
      this.app.workspace.openLinkText(target, this.current.path, 'tab').catch(error => {
        new Notice(`无法打开链接：${error.message || error}`);
      });
    });
    this.message('正在准备知识卡片…');
  }
  message(text) {
    this.reading.empty();
    this.reading.createEl('p', { text, cls: 'wkc-message' });
  }
  async next() {
    const token = ++this.request;
    this.current = null;
    if (this.renderer) { this.removeChild(this.renderer); this.renderer = null; }
    this.openButton.disabled = true;
    this.message('正在翻开下一张…');
    let renderer;
    try {
      const path = await this.plugin.pick();
      if (token !== this.request) return;
      if (!path) { this.message('还没有可展示的页面。请在 wiki 的概念、模型或比较目录中添加笔记。'); return; }
      const file = this.app.vault.getAbstractFileByPath(path);
      if (!file) throw new Error('原文已移走或删除，请换一张。');
      const text = await this.app.vault.read(file);
      if (token !== this.request) return;
      const card = extractCard(text, path, this.app.metadataCache.getFileCache(file)?.frontmatter || {});
      const staging = document.createElement('div');
      staging.className = 'wkc-content';
      staging.createDiv({ text: TYPES[card.kind], cls: 'wkc-kind' });
      staging.createEl('h1', { text: card.title, cls: 'wkc-title' });
      renderer = new Component();
      this.addChild(renderer);
      if (card.description) {
        const summary = staging.createDiv({ cls: 'wkc-summary' });
        await MarkdownRenderer.render(this.app, card.description, summary, path, renderer);
      }
      if (card.heading) staging.createEl('h2', { text: card.heading, cls: 'wkc-heading' });
      const body = staging.createDiv({ cls: 'wkc-body markdown-rendered' });
      await MarkdownRenderer.render(this.app, card.body, body, path, renderer);
      staging.createDiv({ text: path, cls: 'wkc-source' });
      if (token !== this.request) { this.removeChild(renderer); return; }
      if (this.renderer) this.removeChild(this.renderer);
      this.renderer = renderer;
      this.reading.replaceChildren(staging);
      this.reading.scrollTop = 0;
      this.current = card;
      this.openButton.disabled = false;
    } catch (error) {
      if (renderer && renderer !== this.renderer) this.removeChild(renderer);
      if (token === this.request) this.message(`暂时无法展示：${error.message || error}`);
    }
  }
  async openOriginal() {
    const card = this.current;
    if (!card) return;
    const file = this.app.vault.getAbstractFileByPath(card.path);
    if (!file) { new Notice('原文已移走或删除，请换一张。'); return; }
    try {
      const leaf = this.app.workspace.getLeavesOfType('markdown').find(l => l.view.file?.path === card.path)
        || this.app.workspace.getLeaf('tab');
      await leaf.openFile(file, { active: true, state: { mode: 'preview' }, eState: { line: card.line } });
      await this.app.workspace.revealLeaf(leaf);
    } catch (error) { new Notice(`无法打开原文：${error.message || error}`); }
  }
  async onClose() {
    ++this.request;
    if (this.renderer) { this.removeChild(this.renderer); this.renderer = null; }
  }
}

class KnowledgeCardsPlugin extends Plugin {
  async onload() {
    this.stopped = false;
    this.queue = Promise.resolve();
    try { this.progress = await this.loadData() || {}; }
    catch { this.progress = {}; new Notice('知识卡片进度无法读取，已开始新一轮。'); }
    this.registerView(VIEW, leaf => new CardView(leaf, this));
    this.addCommand({ id: 'open-cards', name: '打开知识卡片', callback: () => this.show(false) });
    this.app.workspace.onLayoutReady(() => {
      const ready = () => {
        if (this.stopped || this.started) return;
        this.started = true;
        this.show(true);
      };
      const files = this.app.vault.getMarkdownFiles().filter(f => category(f.path));
      if (files.every(f => this.app.metadataCache.getFileCache(f))) ready();
      else {
        this.registerEvent(this.app.metadataCache.on('resolved', ready));
        // 异常笔记未产生缓存时仍允许以正文标题降级展示。
        const timer = setTimeout(ready, 5000);
        this.register(() => clearTimeout(timer));
      }
    });
  }
  pick() {
    const task = this.queue.then(async () => {
      const paths = this.app.vault.getMarkdownFiles().filter(f => category(f.path)).map(f => f.path);
      const result = draw(paths, this.progress);
      this.progress = result.state;
      await this.saveData(this.progress);
      return result.path;
    });
    this.queue = task.catch(() => {});
    return task;
  }
  async show(refresh) {
    if (this.opening) return this.opening;
    this.opening = (async () => {
      try {
        let leaf = this.app.workspace.getLeavesOfType(VIEW)[0];
        if (!leaf) {
          leaf = this.app.workspace.getLeaf('tab');
          await leaf.setViewState({ type: VIEW, active: true });
        }
        if (this.stopped) return;
        await this.app.workspace.revealLeaf(leaf);
        if (refresh || !leaf.view.current) await leaf.view.next();
      } catch (error) { new Notice(`知识卡片打开失败：${error.message || error}`); }
    })();
    try { await this.opening; } finally { this.opening = null; }
  }
  onunload() {
    this.stopped = true;
    this.app.workspace.getLeavesOfType(VIEW).forEach(leaf => leaf.detach());
  }
}

module.exports = KnowledgeCardsPlugin;
module.exports.testing = { extractCard, draw, category };
