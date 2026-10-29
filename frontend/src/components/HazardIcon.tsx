import type { CSSProperties } from "react";
import { hazardDefinition } from "../utils/hazardMeta";

export function HazardIcon({
  type,
  size = 18,
  className = "",
}: {
  type: string | null | undefined;
  size?: number;
  className?: string;
}) {
  const hazard = hazardDefinition(type);
  const style = { "--hazard-color": hazard.color } as CSSProperties;
  return (
    <svg
      className={`hazard-svg ${className}`}
      style={style}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={hazard.path} />
    </svg>
  );
}
