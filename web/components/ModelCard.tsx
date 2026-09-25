"use client";
import { useEffect, useState } from "react";
import { modelInfo } from "@/lib/model";

type Card = {
  name?: string;
  arch?: string;
  params_m?: number;
  size_mb?: number;
  input?: string;
  trained_on?: string;
  iterations?: number;
  val?: { mIoU: number; pixel_acc: number; per_class_iou: Record<string, number> };
  notes?: string[];
  precision?: string;
  vectorization?: {
    plans: number;
    rooms: { f1: number; precision: number; recall: number; matched_mean_iou: number; exact_count_rate: number };
    doors: { f1: number };
    windows: { f1: number };
  };
};

export default function ModelCard() {
  const [card, setCard] = useState<Card | null>(null);
  useEffect(() => {
    modelInfo().then((c) => setCard(c as Card | null));
  }, []);
  if (!card) return <p className="muted">Model card unavailable.</p>;
  return (
    <div className="card-grid">
      <div className="card">
        <h4>Browser model</h4>
        <dl className="kv">
          <dt>Architecture</dt><dd>{card.arch}</dd>
          <dt>Parameters</dt><dd>{card.params_m?.toFixed(2)} M</dd>
          <dt>ONNX size</dt><dd>{card.size_mb?.toFixed(1)} MB{card.precision ? ` (${card.precision})` : ""}</dd>
          <dt>Input</dt><dd>{card.input}</dd>
          <dt>Trained on</dt><dd>{card.trained_on}</dd>
          <dt>Iterations</dt><dd>{card.iterations?.toLocaleString()}</dd>
        </dl>
      </div>
      {card.val && (
        <div className="card">
          <h4>Validation (held-out synthetic plans)</h4>
          <p className="big">{(card.val.mIoU * 100).toFixed(1)}<small> mIoU</small></p>
          <div className="bars">
            {Object.entries(card.val.per_class_iou).map(([k, v]) => (
              <div key={k} className="barrow">
                <span>{k}</span>
                <div className="bar"><i style={{ width: `${(v * 100).toFixed(1)}%` }} /></div>
                <span className="num">{(v * 100).toFixed(1)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {card.vectorization && (
        <div className="card">
          <h4>Vector output ({card.vectorization.plans} held-out plans)</h4>
          <div className="bars">
            {[
              ["rooms F1", card.vectorization.rooms.f1],
              ["room IoU", card.vectorization.rooms.matched_mean_iou],
              ["doors F1", card.vectorization.doors.f1],
              ["windows F1", card.vectorization.windows.f1],
              ["exact count", card.vectorization.rooms.exact_count_rate],
            ].map(([k, v]) => (
              <div key={k as string} className="barrow">
                <span>{k}</span>
                <div className="bar"><i style={{ width: `${((v as number) * 100).toFixed(1)}%` }} /></div>
                <span className="num">{((v as number) * 100).toFixed(1)}</span>
              </div>
            ))}
          </div>
          <p className="muted small">Rooms matched at polygon IoU 0.5; openings matched by centre distance. Measured by scripts/eval_vectorization.py.</p>
        </div>
      )}
      {card.notes && (
        <div className="card">
          <h4>Read before you rely on it</h4>
          <ul className="notes">{card.notes.map((n) => <li key={n}>{n}</li>)}</ul>
        </div>
      )}
    </div>
  );
}
