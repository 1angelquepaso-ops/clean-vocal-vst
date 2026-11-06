//! Clean Vocal panel: three rotary knobs (Strength, Hum Cut Freq) plus two switches
//! (Extra Clean, Hum Cut), laid out as four controls in a row.
//!
//! Knobs: drag up/down to turn (hold Shift for fine control), double-click to reset.

use crate::CleanVocalParams;
use nih_plug::prelude::*;
use nih_plug_egui::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, Response, Sense, Stroke, Ui, Vec2,
};
use nih_plug_egui::{create_egui_editor, EguiState};
use std::f32::consts::PI;
use std::sync::Arc;

pub const WIDTH: u32 = 520;
pub const HEIGHT: u32 = 230;

pub fn default_state() -> Arc<EguiState> {
    EguiState::from_size(WIDTH, HEIGHT)
}

// Palette
const BG: Color32 = Color32::from_rgb(0x16, 0x1A, 0x1F);
const PANEL: Color32 = Color32::from_rgb(0x20, 0x26, 0x2D);
const TRACK: Color32 = Color32::from_rgb(0x34, 0x3C, 0x46);
const ACCENT: Color32 = Color32::from_rgb(0x2E, 0xD3, 0xA0);
const ACCENT_DIM: Color32 = Color32::from_rgb(0x1B, 0x6E, 0x56);
const TEXT: Color32 = Color32::from_rgb(0xE8, 0xEC, 0xF0);
const TEXT_DIM: Color32 = Color32::from_rgb(0x8A, 0x94, 0xA0);

const KNOB_START: f32 = 0.75 * PI; // 7:30 o'clock
const KNOB_SWEEP: f32 = 1.5 * PI; // 270 degrees

pub fn create(params: Arc<CleanVocalParams>) -> Option<Box<dyn Editor>> {
    let state = params.editor_state.clone();
    create_egui_editor(
        state,
        (),
        |ctx, _| {
            let mut style = (*ctx.style()).clone();
            style.visuals.panel_fill = BG;
            style.visuals.window_fill = BG;
            ctx.set_style(style);
        },
        move |ctx, setter, _| {
            egui::CentralPanel::default()
                .frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin::same(16)))
                .show(ctx, |ui| {
                    header(ui);
                    ui.add_space(10.0);
                    let row_h = ui.available_height();
                    let cell_w = (ui.available_width() - 3.0 * 10.0) / 4.0;
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 10.0;
                        let hum_on = params.hum_cut.value();
                        cell(ui, cell_w, row_h, "STRENGTH", true, |ui| {
                            knob(ui, &params.strength, setter, true)
                        });
                        cell(ui, cell_w, row_h, "EXTRA CLEAN", true, |ui| {
                            switch(ui, &params.extra_clean, setter)
                        });
                        cell(ui, cell_w, row_h, "HUM CUT", true, |ui| {
                            switch(ui, &params.hum_cut, setter)
                        });
                        cell(ui, cell_w, row_h, "HUM FREQ", hum_on, |ui| {
                            knob(ui, &params.hum_cut_hz, setter, hum_on)
                        });
                    });
                });
        },
    )
}

fn header(ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("CLEAN VOCAL").color(TEXT).size(18.0).strong());
        ui.label(egui::RichText::new("AI noise removal").color(TEXT_DIM).size(12.0));
    });
}

/// A rounded card with a caption at the bottom; `content` draws the control centred in it.
fn cell(
    ui: &mut Ui,
    w: f32,
    h: f32,
    caption: &str,
    active: bool,
    content: impl FnOnce(&mut Ui) -> String,
) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(w, h), Sense::hover());
    ui.painter().rect_filled(rect, 10.0, PANEL);
    let control_rect = Rect::from_min_max(rect.min, Pos2::new(rect.max.x, rect.max.y - 44.0));
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(control_rect)
            .layout(egui::Layout::centered_and_justified(egui::Direction::TopDown)),
    );
    let value_text = content(&mut child);
    let p = ui.painter();
    p.text(
        Pos2::new(rect.center().x, rect.max.y - 34.0),
        Align2::CENTER_CENTER,
        value_text,
        FontId::proportional(14.0),
        if active { TEXT } else { TEXT_DIM },
    );
    p.text(
        Pos2::new(rect.center().x, rect.max.y - 14.0),
        Align2::CENTER_CENTER,
        caption,
        FontId::proportional(11.0),
        TEXT_DIM,
    );
}

/// Rotary knob bound to a parameter. Returns the value text to show under it.
fn knob<P: Param>(ui: &mut Ui, param: &P, setter: &ParamSetter, active: bool) -> String {
    let size = ui.available_size().min_elem().min(96.0);
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(size), Sense::click_and_drag());
    handle_knob_input(ui, &response, param, setter);

    let value = param.modulated_normalized_value();
    let c = rect.center();
    let r = size * 0.42;
    let p = ui.painter();
    let col = if active { ACCENT } else { ACCENT_DIM };

    // Track and value arcs
    arc(p, c, r, KNOB_START, KNOB_START + KNOB_SWEEP, Stroke::new(6.0_f32, TRACK));
    if value > 0.001 {
        arc(p, c, r, KNOB_START, KNOB_START + KNOB_SWEEP * value, Stroke::new(6.0_f32, col));
    }
    // Knob body + pointer
    p.circle_filled(c, r * 0.72, Color32::from_rgb(0x2B, 0x32, 0x3B));
    p.circle_stroke(c, r * 0.72, Stroke::new(1.0_f32, TRACK));
    let a = KNOB_START + KNOB_SWEEP * value;
    let dir = Vec2::new(a.cos(), a.sin());
    p.line_segment([c + dir * r * 0.25, c + dir * r * 0.62], Stroke::new(3.0_f32, if active { TEXT } else { TEXT_DIM }));

    if response.hovered() || response.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
    }
    param.to_string()
}

fn handle_knob_input<P: Param>(ui: &Ui, response: &Response, param: &P, setter: &ParamSetter) {
    if response.double_clicked() {
        setter.begin_set_parameter(param);
        setter.set_parameter(param, param.default_plain_value());
        setter.end_set_parameter(param);
        return;
    }
    if response.drag_started() {
        setter.begin_set_parameter(param);
    }
    if response.dragged() {
        let fine = ui.input(|i| i.modifiers.shift);
        let speed = if fine { 0.0012 } else { 0.006 };
        let delta = -response.drag_delta().y * speed;
        if delta != 0.0 {
            let v = (param.unmodulated_normalized_value() + delta).clamp(0.0, 1.0);
            setter.set_parameter_normalized(param, v);
        }
    }
    if response.drag_stopped() {
        setter.end_set_parameter(param);
    }
}

/// On/off switch bound to a bool parameter. Returns "ON"/"OFF".
fn switch(ui: &mut Ui, param: &BoolParam, setter: &ParamSetter) -> String {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(64.0, 34.0), Sense::click());
    let on = param.value();
    if response.clicked() {
        setter.begin_set_parameter(param);
        setter.set_parameter(param, !on);
        setter.end_set_parameter(param);
    }
    let on = param.value();
    let p = ui.painter();
    p.rect_filled(rect, 17.0, if on { ACCENT } else { TRACK });
    let x = if on { rect.max.x - 17.0 } else { rect.min.x + 17.0 };
    p.circle_filled(Pos2::new(x, rect.center().y), 13.0, TEXT);
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    if on { "ON".into() } else { "OFF".into() }
}

fn arc(p: &egui::Painter, c: Pos2, r: f32, a0: f32, a1: f32, stroke: Stroke) {
    let n = (((a1 - a0).abs() / (2.0 * PI)) * 64.0).ceil().max(2.0) as usize;
    let pts: Vec<Pos2> = (0..=n)
        .map(|i| {
            let a = a0 + (a1 - a0) * i as f32 / n as f32;
            c + Vec2::new(a.cos(), a.sin()) * r
        })
        .collect();
    p.add(egui::Shape::line(pts, stroke));
}
