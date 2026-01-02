import { getBackendURL } from "./baseURL";

export async function apiFetch(path, options = {}) {
  const base = await getBackendURL();
  const url = `${base}${path.startsWith("/") ? path : `/${path}`}`;
  return fetch(url, options);
}
