#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Unknown = 0,
    Yolo = 1,
    PaddleOcr = 2,
    OpenVino = 3,
    OpenVinoOcr = 4,
}
