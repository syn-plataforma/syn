/* eslint-disable @typescript-eslint/naming-convention */

/**
 * Clase para formar el RequestBody de la petición de login
 */
export class LoginRequest {
  constructor(private username: string, private password: string) {}

  transform() {
    return {
      username: this.username,
      password: this.password,
    };
  }
}
/**
 * Interfaz para la respuesta de la petición de login
 */
export interface LoginResponse {
  access_token: string;
  code: string;
  message: string;
}
