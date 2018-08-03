import { Injectable } from '@angular/core';
import { CanActivate } from '@angular/router';
import { AuthenticationService } from '@kognifai/poseidon-ng-authenticationservice';

@Injectable()
export class AuthenticationGuardService implements CanActivate {
  constructor(private authenticationService: AuthenticationService) { }

  canActivate(): boolean {
    return this.authenticationService.isAuthenticated;
  }
}
