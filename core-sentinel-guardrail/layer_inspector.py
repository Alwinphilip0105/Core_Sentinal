import html
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets


# ── Color palette (matches existing app dark theme) ──
_C = {
    "regex": "#3B82F6",  # blue
    "ner": "#8B5CF6",  # purple
    "ml": "#10B981",  # green
    "multi": "#F59E0B",  # amber — multiple layers
    "critical": "#EF4444",  # red — critical secret
    "bg": "#111827",
    "surface": "#1E2130",
    "border": "#2D3748",
    "text": "#F1F5F9",
    "muted": "#9CA3AF",
}

# Highlight backgrounds (dark variants for readability)
_HL = {
    "regex": "#1E3A5F",
    "kb_rule": "#1E3A5F",
    "ner": "#2D1B5E",
    "model_window": "#064E3B",
    "ml": "#064E3B",
    "critical_secret": "#7F1D1D",
    "multi": "#451A03",
}


def _norm_src(s: str) -> str:
    """Normalise raw span source to display category."""
    s = str(s or "").lower()
    if s in ("regex", "kb_rule"):
        return "regex"
    if s in ("model_window", "ml"):
        return "ml"
    if s == "critical_secret":
        return "critical_secret"
    if s == "ner":
        return "ner"
    return s


class LayerInspectorPage(QtWidgets.QWidget):
    """
    Standalone top-level window.
    Shows how Regex / NER / ML layers each contribute
    to PII detection on a given text.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(
            parent,
            QtCore.Qt.WindowType.Window,
        )
        self.setWindowTitle(
            "Layer Inspector — Core Sentinel"
        )
        self.resize(940, 660)
        self.setMinimumSize(720, 500)
        self._result: dict = {}
        self._spans: list = []
        self._text: str = ""
        self._apply_qss()
        self._build_ui()

    # ── QSS ─────────────────────────────────────────
    def _apply_qss(self) -> None:
        self.setStyleSheet(f"""
          QWidget {{
            background: {_C['bg']};
            color: {_C['text']};
            font-family: 'Segoe UI', sans-serif;
            font-size: 13px;
          }}
          QFrame#pipelineBox {{
            background: {_C['surface']};
            border: 1px solid {_C['border']};
            border-radius: 8px;
            padding: 6px 10px;
          }}
          QFrame#layerCard {{
            background: {_C['surface']};
            border: 1px solid {_C['border']};
            border-radius: 8px;
          }}
          QFrame#detailPanel {{
            background: {_C['surface']};
            border: 1px solid {_C['border']};
            border-radius: 8px;
          }}
          QPlainTextEdit {{
            background: #0F172A;
            border: 1px solid {_C['border']};
            border-radius: 8px;
            padding: 8px;
            font-family: Consolas, 'Courier New',
                         monospace;
            font-size: 12px;
            color: {_C['text']};
          }}
          QPlainTextEdit:focus {{
            border-color: {_C['regex']};
          }}
          QTextEdit {{
            background: #0F172A;
            border: 1px solid {_C['border']};
            border-radius: 8px;
            padding: 8px;
            font-family: Consolas, 'Courier New',
                         monospace;
            font-size: 12px;
          }}
          QPushButton#analyzeBtn {{
            background: #2563EB;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 9px;
            font-size: 13px;
            font-weight: 600;
          }}
          QPushButton#analyzeBtn:hover {{
            background: #1D4ED8;
          }}
          QPushButton#analyzeBtn:pressed {{
            background: #1E40AF;
          }}
          QScrollBar:vertical {{
            background: {_C['bg']};
            width: 6px;
            border-radius: 3px;
          }}
          QScrollBar::handle:vertical {{
            background: {_C['border']};
            border-radius: 3px;
            min-height: 20px;
          }}
        """)

    # ── UI build ─────────────────────────────────────
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        # Section A — pipeline flow
        root.addWidget(self._build_pipeline_section())

        # Section B — analysis area
        root.addLayout(
            self._build_analysis_section(), stretch=1
        )

        # Section C — per-layer metric cards
        root.addWidget(self._build_cards_section())

    # ── Section A: pipeline diagram ──────────────────
    def _build_pipeline_section(
        self,
    ) -> QtWidgets.QWidget:
        wrap = QtWidgets.QFrame()
        wrap.setObjectName("pipelineBox")
        wrap.setFixedHeight(84)
        lay = QtWidgets.QHBoxLayout(wrap)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)

        boxes = [
            ("Input", None, "text"),
            ("Regex", _C["regex"], "spans"),
            ("NER", _C["ner"], "spans"),
            ("ML Model", _C["ml"], "spans"),
            ("Output", "#4B5563", "decision"),
        ]

        self._pipe_boxes: dict[str, dict] = {}

        for i, (name, accent, kind) in enumerate(boxes):
            if i > 0:
                arr = QtWidgets.QLabel("›")
                arr.setStyleSheet(
                    "color:#4B5563;font-size:18px;"
                    "background:transparent;"
                )
                arr.setAlignment(
                    QtCore.Qt.AlignmentFlag.AlignCenter
                )
                lay.addWidget(arr)

            box = QtWidgets.QFrame()
            box.setObjectName("pipelineBox")
            border_style = (
                f"border-left:3px solid {accent};"
                if accent else ""
            )
            box.setStyleSheet(
                f"QFrame#pipelineBox{{"
                f"background:{_C['surface']};"
                f"border:1px solid {_C['border']};"
                f"border-radius:8px;"
                f"{border_style}}}"
            )
            box.setMinimumWidth(100)

            blay = QtWidgets.QVBoxLayout(box)
            blay.setContentsMargins(8, 4, 8, 4)
            blay.setSpacing(2)

            name_lbl = QtWidgets.QLabel(name)
            name_lbl.setStyleSheet(
                "font-weight:600;font-size:12px;"
                f"color:{accent or _C['muted']};"
                "background:transparent;"
            )

            val_lbl = QtWidgets.QLabel("—")
            val_lbl.setStyleSheet(
                f"color:{_C['muted']};font-size:11px;"
                "background:transparent;"
            )

            blay.addWidget(name_lbl)
            blay.addWidget(val_lbl)
            lay.addWidget(box, stretch=1)

            self._pipe_boxes[name] = {
                "frame": box,
                "val": val_lbl,
                "accent": accent,
            }

        return wrap

    # ── Section B: analysis area ─────────────────────
    def _build_analysis_section(
        self,
    ) -> QtWidgets.QHBoxLayout:
        lay = QtWidgets.QHBoxLayout()
        lay.setSpacing(12)
        lay.addLayout(
            self._build_input_panel(), stretch=6
        )
        lay.addWidget(
            self._build_detail_panel(), stretch=4
        )
        return lay

    def _build_input_panel(
        self,
    ) -> QtWidgets.QVBoxLayout:
        lay = QtWidgets.QVBoxLayout()
        lay.setSpacing(8)

        lbl = QtWidgets.QLabel("Paste text to inspect:")
        lbl.setStyleSheet(
            f"color:{_C['muted']};font-size:12px;"
        )
        lay.addWidget(lbl)

        self._input = QtWidgets.QPlainTextEdit()
        self._input.setPlaceholderText(
            "Paste any text here to see which layer "
            "detects PII and why…"
        )
        self._input.setFixedHeight(90)
        self._input.textChanged.connect(
            self._on_input_changed
        )
        lay.addWidget(self._input)

        btn_row = QtWidgets.QHBoxLayout()
        self._analyze_btn = QtWidgets.QPushButton(
            "Analyze"
        )
        self._analyze_btn.setObjectName("analyzeBtn")
        self._analyze_btn.clicked.connect(
            self._run_analysis
        )
        self._char_lbl = QtWidgets.QLabel("0 characters")
        self._char_lbl.setStyleSheet(
            f"color:{_C['muted']};font-size:11px;"
        )
        btn_row.addWidget(self._analyze_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._char_lbl)
        lay.addLayout(btn_row)

        results_lbl = QtWidgets.QLabel("Results:")
        results_lbl.setStyleSheet(
            f"color:{_C['muted']};font-size:12px;"
        )
        lay.addWidget(results_lbl)

        self._results_view = QtWidgets.QTextEdit()
        self._results_view.setReadOnly(True)
        self._results_view.setPlaceholderText(
            "Highlighted results appear here…"
        )
        self._results_view.mousePressEvent = (
            self._on_results_click
        )
        lay.addWidget(self._results_view, stretch=1)

        # Legend
        legend = QtWidgets.QHBoxLayout()
        legend.setSpacing(12)
        for label, color in [
            ("Regex", _C["regex"]),
            ("NER", _C["ner"]),
            ("ML Model", _C["ml"]),
            ("Multi-layer", _C["multi"]),
            ("Critical", _C["critical"]),
        ]:
            dot = QtWidgets.QLabel("■")
            dot.setStyleSheet(
                f"color:{color};font-size:14px;"
            )
            txt = QtWidgets.QLabel(label)
            txt.setStyleSheet(
                f"color:{_C['muted']};font-size:11px;"
            )
            legend.addWidget(dot)
            legend.addWidget(txt)
        legend.addStretch(1)
        lay.addLayout(legend)

        return lay

    def _build_detail_panel(
        self,
    ) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame()
        panel.setObjectName("detailPanel")
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)

        title = QtWidgets.QLabel("Detection Detail")
        title.setStyleSheet(
            "font-size:14px;font-weight:600;"
            f"color:{_C['text']};"
        )
        lay.addWidget(title)

        self._detail_hint = QtWidgets.QLabel(
            "Click a highlighted span for detail"
        )
        self._detail_hint.setStyleSheet(
            f"color:{_C['muted']};font-size:12px;"
        )
        self._detail_hint.setWordWrap(True)
        lay.addWidget(self._detail_hint)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(
            QtWidgets.QFrame.Shape.HLine
        )
        sep.setStyleSheet(
            f"color:{_C['border']};"
        )
        lay.addWidget(sep)

        # Span info (hidden until clicked)
        self._detail_widget = QtWidgets.QWidget()
        self._detail_widget.hide()
        dlay = QtWidgets.QVBoxLayout(self._detail_widget)
        dlay.setContentsMargins(0, 0, 0, 0)
        dlay.setSpacing(6)

        self._detail_text = QtWidgets.QLabel()
        self._detail_text.setStyleSheet(
            "font-family:Consolas,'Courier New',"
            f"monospace;font-size:13px;color:{_C['text']};"
            f"background:#0F172A;border-radius:4px;"
            "padding:4px 8px;"
        )
        self._detail_text.setWordWrap(True)
        dlay.addWidget(self._detail_text)

        self._detail_category = QtWidgets.QLabel()
        self._detail_category.setStyleSheet(
            "font-size:11px;font-weight:600;"
            "padding:2px 8px;border-radius:4px;"
        )
        dlay.addWidget(self._detail_category)

        dlay.addSpacing(4)
        detected_lbl = QtWidgets.QLabel("Detected by:")
        detected_lbl.setStyleSheet(
            f"color:{_C['muted']};font-size:12px;"
            "font-weight:600;"
        )
        dlay.addWidget(detected_lbl)

        # Layer rows
        self._layer_rows: dict[str, QtWidgets.QLabel] = {}
        for layer, color in [
            ("Regex", _C["regex"]),
            ("NER", _C["ner"]),
            ("ML Model", _C["ml"]),
        ]:
            row_lbl = QtWidgets.QLabel()
            row_lbl.setStyleSheet(
                f"font-size:12px;color:{_C['muted']};"
            )
            dlay.addWidget(row_lbl)
            self._layer_rows[layer] = row_lbl

        dlay.addSpacing(4)
        self._detail_prob = QtWidgets.QLabel()
        self._detail_prob.setStyleSheet(
            f"font-size:12px;color:{_C['muted']};"
        )
        dlay.addWidget(self._detail_prob)

        dlay.addStretch(1)
        lay.addWidget(self._detail_widget)
        lay.addStretch(1)

        return panel

    # ── Section C: layer metric cards ────────────────
    def _build_cards_section(
        self,
    ) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self._cards: dict[str, dict] = {}
        defs = [
            ("Regex", _C["regex"],
             "Pattern-matched PII"),
            ("NER", _C["ner"],
             "Named entity recognition"),
            ("ML Model", _C["ml"],
             "Neural classifier"),
        ]

        for name, color, desc in defs:
            card = QtWidgets.QFrame()
            card.setObjectName("layerCard")
            card.setStyleSheet(
                f"QFrame#layerCard{{"
                f"background:{_C['surface']};"
                f"border:1px solid {_C['border']};"
                f"border-left:3px solid {color};"
                f"border-radius:8px;}}"
            )
            clay = QtWidgets.QVBoxLayout(card)
            clay.setContentsMargins(12, 10, 12, 10)
            clay.setSpacing(3)

            n_lbl = QtWidgets.QLabel("—")
            n_lbl.setStyleSheet(
                f"font-size:22px;font-weight:700;"
                f"color:{color};"
            )
            t_lbl = QtWidgets.QLabel(name)
            t_lbl.setStyleSheet(
                f"font-size:12px;font-weight:600;"
                f"color:{_C['text']};"
            )
            d_lbl = QtWidgets.QLabel(desc)
            d_lbl.setStyleSheet(
                f"font-size:11px;color:{_C['muted']};"
            )
            sub_lbl = QtWidgets.QLabel("")
            sub_lbl.setStyleSheet(
                f"font-size:11px;color:{_C['muted']};"
            )

            clay.addWidget(n_lbl)
            clay.addWidget(t_lbl)
            clay.addWidget(d_lbl)
            clay.addWidget(sub_lbl)
            lay.addWidget(card, stretch=1)

            self._cards[name] = {
                "count": n_lbl,
                "sub": sub_lbl,
            }

        return wrap

    # ── Slots ────────────────────────────────────────
    def _on_input_changed(self) -> None:
        n = len(self._input.toPlainText())
        self._char_lbl.setText(f"{n} characters")

    def _run_analysis(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            self._show_toast("Enter some text first.")
            return

        self._analyze_btn.setEnabled(False)
        self._analyze_btn.setText("Analyzing…")
        QtWidgets.QApplication.processEvents()

        try:
            from infer import score_clipboard_with_pii
            result = score_clipboard_with_pii(text)
        except Exception as exc:
            self._show_toast(f"Error: {exc}")
            self._analyze_btn.setEnabled(True)
            self._analyze_btn.setText("Analyze")
            return

        self._text = text
        self._result = result
        self._spans = result.get("spans") or []

        self._results_view.setHtml(
            self._build_highlight_html(text, self._spans)
        )
        self._update_pipeline_boxes(result)
        self._update_layer_cards(result)

        self._analyze_btn.setEnabled(True)
        self._analyze_btn.setText("Analyze")

    def _on_results_click(
        self, event: QtGui.QMouseEvent
    ) -> None:
        # Let QTextEdit handle the event normally first
        QtWidgets.QTextEdit.mousePressEvent(
            self._results_view, event
        )
        cursor = self._results_view.cursorForPosition(
            event.pos()
        )
        pos = cursor.position()
        # Map HTML char position back to text offset
        # (rough approximation: strip tags for offset)
        text = self._text
        for span in self._spans:
            start = span.get("start", 0)
            end = span.get("end", 0)
            if start <= pos < end:
                self._show_span_detail(span)
                return
        self._detail_widget.hide()
        self._detail_hint.show()

    def _show_span_detail(self, span: dict) -> None:
        # span keys: class/label, match/text, source,
        #            score (optional), risk (optional)
        span_text = (
            span.get("text")
            or span.get("match")
            or ""
        )
        category = (
            span.get("label")
            or span.get("class")
            or "Unknown"
        )
        source = str(span.get("source", "")).lower()
        score = span.get("score")
        self._detail_text.setText(
            f'"{span_text}"'
        )

        # Category badge
        norm = _norm_src(source)
        badge_color = _C.get(norm, _C["critical"] if norm == "critical_secret" else _C["muted"])
        self._detail_category.setText(str(category))
        self._detail_category.setStyleSheet(
            f"font-size:11px;font-weight:600;"
            f"padding:2px 8px;border-radius:4px;"
            f"background:{badge_color}22;"
            f"color:{badge_color};"
        )

        # Layer checkmarks
        detected_by = {
            "Regex": norm in ("regex",),
            "NER": norm == "ner",
            "ML Model": norm in ("ml", "multi"),
        }
        muted = _C["muted"]
        for layer, active in detected_by.items():
            icon = "✓" if active else "○"
            col = _C["ml"] if active else _C["muted"]
            note = ""
            if active:
                if layer == "Regex":
                    note = f"  {category} pattern"
                elif layer == "NER":
                    note = f"  {category} entity"
                elif layer == "ML Model" and score is not None:
                    try:
                        note = f"  p = {float(score):.3f}"
                    except (TypeError, ValueError):
                        note = ""
            self._layer_rows[layer].setTextFormat(
                QtCore.Qt.TextFormat.RichText
            )
            self._layer_rows[layer].setText(
                f'<span style="color:{col}">'
                f"{icon} {layer}</span>"
                f'<span style="color:{muted}">'
                f"{note}</span>"
            )

        if score is not None:
            try:
                self._detail_prob.setText(
                    f"ML confidence: p = {float(score):.3f}"
                )
            except (TypeError, ValueError):
                self._detail_prob.setText("ML confidence: (n/a)")
        elif self._result.get("prob_risky") is not None:
            pr = self._result["prob_risky"]
            self._detail_prob.setText(
                "P(risky) = "
                f"{float(pr):.3f}"
            )
        else:
            self._detail_prob.setText(
                "Critical path — ML not run"
            )

        self._detail_hint.hide()
        self._detail_widget.show()

    # ── Pipeline box updates ─────────────────────────
    def _update_pipeline_boxes(
        self, result: dict
    ) -> None:
        ls = result.get("layer_summary") or {}
        decision = str(result.get("decision", "?"))
        prob = result.get("prob_risky")

        ml_suffix = (
            f"  p={prob:.2f}" if prob is not None else "  (critical path)"
        )
        updates = {
            "Regex": str(ls.get("regex_count", 0))
            + " spans",
            "NER": str(ls.get("ner_count", 0))
            + " spans",
            "ML Model": (
                str(ls.get("ml_count", 0))
                + " spans"
                + ml_suffix
            ),
            "Output": decision.upper(),
        }

        dec_colors = {
            "block": "#EF4444",
            "warn": "#F59E0B",
            "allow": "#10B981",
        }
        out_color = dec_colors.get(
            decision.lower(), "#4B5563"
        )

        for name, text in updates.items():
            entry = self._pipe_boxes.get(name)
            if entry:
                entry["val"].setText(text)
                if name == "Output":
                    entry["val"].setStyleSheet(
                        f"color:{out_color};"
                        "font-weight:700;"
                        "font-size:12px;"
                        "background:transparent;"
                    )

    # ── Layer card updates ───────────────────────────
    def _update_layer_cards(
        self, result: dict
    ) -> None:
        ls = result.get("layer_summary") or {}
        prob = result.get("prob_risky")

        self._cards["Regex"]["count"].setText(
            str(ls.get("regex_count", 0))
        )
        self._cards["NER"]["count"].setText(
            str(ls.get("ner_count", 0))
        )
        self._cards["ML Model"]["count"].setText(
            str(ls.get("ml_count", 0))
        )
        if prob is not None:
            self._cards["ML Model"]["sub"].setText(
                f"P(risky) = {prob:.3f}"
            )
        else:
            self._cards["ML Model"]["sub"].setText(
                "Critical path — model skipped"
            )

    # ── Highlight HTML builder ───────────────────────
    def _build_highlight_html(
        self, text: str, spans: list
    ) -> str:
        if not text:
            return ""
        if not spans:
            return (
                "<pre style='font-family:Consolas,"
                "\"Courier New\",monospace;"
                f"font-size:13px;color:{_C['muted']};"
                "white-space:pre-wrap;line-height:1.7'>"
                + html.escape(text) +
                "</pre>"
            )

        # Build char-level source map
        src_map = [""] * len(text)
        for span in sorted(
            spans,
            key=lambda s: s.get(
                "score", 0.0
            ) or 0.0
        ):
            start = span.get("start", 0)
            end = span.get("end", 0)
            src = _norm_src(
                span.get("source", "")
            )
            for i in range(
                start, min(end, len(text))
            ):
                if not src_map[i]:
                    src_map[i] = src
                elif src_map[i] != src:
                    src_map[i] = "multi"

        # Render HTML
        out = [
            "<pre style='font-family:Consolas,"
            "\"Courier New\",monospace;"
            f"font-size:13px;color:{_C['text']};"
            "white-space:pre-wrap;line-height:1.7'>"
        ]
        i = 0
        while i < len(text):
            src = src_map[i]
            # Find run end
            j = i + 1
            while j < len(text) and src_map[j] == src:
                j += 1
            chunk = html.escape(text[i:j])
            if src:
                bg = _HL.get(src, "#374151")
                tip = src.upper().replace("_", " ")
                out.append(
                    f'<span style="background:{bg};'
                    "border-radius:3px;"
                    'padding:0 2px;cursor:pointer" '
                    f'title="{html.escape(tip)}">{chunk}</span>'
                )
            else:
                out.append(chunk)
            i = j

        out.append("</pre>")
        return "".join(out)

    # ── Public API (called from overlay) ────────────
    def load_result(
        self, result: dict, text: str = ""
    ) -> None:
        """
        Pre-populate from the overlay's last result.
        Pass text separately (not in result dict).
        """
        self._text = text
        self._result = result
        self._spans = result.get("spans") or []

        if text:
            self._input.setPlainText(text)

        if self._spans or text:
            self._results_view.setHtml(
                self._build_highlight_html(
                    text, self._spans
                )
            )

        self._update_pipeline_boxes(result)
        self._update_layer_cards(result)

    # ── Toast (simple status bar message) ───────────
    def _show_toast(self, msg: str) -> None:
        QtWidgets.QToolTip.showText(
            self.mapToGlobal(
                QtCore.QPoint(
                    self.width() // 2,
                    self.height() - 40,
                )
            ),
            msg,
            self,
        )
